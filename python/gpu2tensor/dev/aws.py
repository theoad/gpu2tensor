"""Small AWS CLI wrapper for this checkout's development runners."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / ".local/aws.json"
REGION = "us-east-1"
STACK = "gpu2tensor-dev"


def aws(*args):
    result = subprocess.run(
        ["aws", *args, "--region", REGION, "--output", "json"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout) if result.stdout.strip() else {}


def read_state():
    return json.loads(STATE.read_text())


def save(state):
    STATE.parent.mkdir(exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE)


def initialize():
    if STATE.exists():
        print("Inventory already exists; use gaws status.")
        return
    image_parameters = {
        "cuda": "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.10-ubuntu-24.04/latest/ami-id",
        "trainium": "/aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id",
    }
    state = {"region": REGION, "stack": STACK, "images": {}, "instances": {}}
    for role, parameter in image_parameters.items():
        image_id = aws("ssm", "get-parameter", "--name", parameter)["Parameter"]["Value"]
        image = aws("ec2", "describe-images", "--image-ids", image_id)["Images"][0]
        state["images"][role] = {key: image[key] for key in
                                  ("ImageId", "Name", "OwnerId", "CreationDate", "RootDeviceName", "BlockDeviceMappings")}
        print(role, image_id, image["Name"], flush=True)
    vpcs = aws("ec2", "describe-vpcs", "--filters", "Name=is-default,Values=true")["Vpcs"]
    if len(vpcs) != 1:
        raise RuntimeError("Expected one default VPC; choose networking explicitly.")
    state["vpc"] = vpcs[0]["VpcId"]
    state["subnets"] = {item["AvailabilityZone"]: item["SubnetId"] for item in
                         aws("ec2", "describe-subnets", "--filters", f"Name=vpc-id,Values={state['vpc']}")["Subnets"]
                         if item["MapPublicIpOnLaunch"]}
    template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "Artifacts": {"Type": "AWS::S3::Bucket", "DeletionPolicy": "Retain", "Properties": {
                "PublicAccessBlockConfiguration": {key: True for key in
                    ("BlockPublicAcls", "BlockPublicPolicy", "IgnorePublicAcls", "RestrictPublicBuckets")},
                "BucketEncryption": {"ServerSideEncryptionConfiguration": [{"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]},
            }},
            "Role": {"Type": "AWS::IAM::Role", "Properties": {
                "AssumeRolePolicyDocument": {"Version": "2012-10-17", "Statement": [{
                    "Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]},
                "ManagedPolicyArns": ["arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"],
                "Policies": [{"PolicyName": "Artifacts", "PolicyDocument": {
                    "Version": "2012-10-17", "Statement": [
                        {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": {"Fn::GetAtt": ["Artifacts", "Arn"]}},
                        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": {"Fn::Sub": "${Artifacts.Arn}/*"}},
                    ]}}],
            }},
            "Profile": {"Type": "AWS::IAM::InstanceProfile", "Properties": {"Roles": [{"Ref": "Role"}]}},
            "Network": {"Type": "AWS::EC2::SecurityGroup", "Properties": {
                "GroupDescription": "gpu2tensor development: SSM only, no ingress",
                "VpcId": state["vpc"],
                "SecurityGroupEgress": [{"IpProtocol": "-1", "CidrIp": "0.0.0.0/0"}],
            }},
        },
        "Outputs": {name: {"Value": {"Ref": resource}} for name, resource in
                    (("bucket", "Artifacts"), ("profile", "Profile"), ("group", "Network"))},
    }
    template_path = STATE.parent / "stack.json"
    template_path.write_text(json.dumps(template, indent=2))
    aws("cloudformation", "validate-template", "--template-body", f"file://{template_path}")
    save(state)
    response = aws("cloudformation", "create-stack", "--stack-name", STACK,
                   "--template-body", f"file://{template_path}", "--capabilities", "CAPABILITY_IAM",
                   "--tags", "Key=Project,Value=gpu2tensor")
    print(json.dumps(response))


def status():
    state = read_state()
    stack = aws("cloudformation", "describe-stacks", "--stack-name", STACK)["Stacks"][0]
    print("Stack:", stack["StackStatus"])
    if stack.get("Outputs"):
        state.update({item["OutputKey"]: item["OutputValue"] for item in stack["Outputs"]})
        save(state)
    result = aws("ec2", "describe-instances", "--filters", "Name=tag:Project,Values=gpu2tensor",
                 "--query", "Reservations[].Instances[].{id:InstanceId,type:InstanceType,state:State.Name,az:Placement.AvailabilityZone}")
    print(json.dumps(result, indent=2))
    print(json.dumps(aws("ssm", "describe-instance-information", "--query",
                         "InstanceInformationList[].{id:InstanceId,status:PingStatus}"), indent=2))


def launch(role, zone, instance_type=None):
    instance_type = instance_type or ("g5.xlarge" if role == "cuda" else "trn1.2xlarge")
    allowed = {"g4dn.xlarge", "g5.xlarge", "g6.xlarge", "g5.2xlarge", "g6.2xlarge"} if role == "cuda" else {"trn1.2xlarge"}
    if instance_type not in allowed:
        raise ValueError("Choose a supported development instance type for this role.")
    state = read_state()
    existing = aws("ec2", "describe-instances", "--filters",
                   f"Name=tag:Name,Values=gpu2tensor-{role}",
                   "Name=instance-state-name,Values=pending,running,stopping,stopped",
                   "--query", "Reservations[].Instances[].InstanceId")
    if existing:
        if len(existing) != 1:
            raise RuntimeError("Multiple instances exist for this role; inspect before continuing.")
        state["instances"][role] = existing[0]
        save(state)
        print("Existing instance:", existing[0])
        return
    image = state["images"][role]
    user_data = """#!/bin/bash
set -eu
mkdir -p /opt/gpu2tensor /opt/gpu2tensor/artifacts
cat > /etc/systemd/system/gpu2tensor-stop.service <<'EOF'
[Service]
Type=oneshot
ExecStart=/usr/sbin/shutdown -h now
EOF
cat > /etc/systemd/system/gpu2tensor-stop.timer <<'EOF'
[Timer]
OnBootSec=4h
Unit=gpu2tensor-stop.service
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now gpu2tensor-stop.timer
"""
    root_size = next(mapping["Ebs"]["VolumeSize"] for mapping in image["BlockDeviceMappings"]
                     if mapping["DeviceName"] == image["RootDeviceName"])
    request = {
        "ImageId": image["ImageId"], "InstanceType": instance_type,
        "MinCount": 1, "MaxCount": 1, "ClientToken": str(uuid.uuid4()),
        "InstanceMarketOptions": {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}},
        "IamInstanceProfile": {"Name": state["profile"]},
        "InstanceInitiatedShutdownBehavior": "terminate",
        "MetadataOptions": {"HttpTokens": "required", "HttpPutResponseHopLimit": 1},
        "NetworkInterfaces": [{"DeviceIndex": 0, "AssociatePublicIpAddress": True, "DeleteOnTermination": True,
                               "SubnetId": state["subnets"].get(zone), "Groups": [state["group"]]}],
        "BlockDeviceMappings": [{"DeviceName": image["RootDeviceName"], "Ebs": {
            "VolumeSize": max(150, root_size), "VolumeType": "gp3", "Encrypted": True, "DeleteOnTermination": True}}],
        "UserData": base64.b64encode(user_data.encode()).decode(),
        "TagSpecifications": [{"ResourceType": kind, "Tags": [
            {"Key": "Project", "Value": "gpu2tensor"}, {"Key": "Name", "Value": f"gpu2tensor-{role}"}]} for kind in ("instance", "volume")],
    }
    if zone == "auto":
        request["NetworkInterfaces"][0].pop("SubnetId")
    path = STATE.parent / f"launch-{role}.json"
    path.write_text(json.dumps(request, indent=2))
    response = aws("ec2", "run-instances", "--cli-input-json", f"file://{path}")
    instance = response["Instances"][0]
    state["instances"][role] = instance["InstanceId"]
    save(state)
    print(role, instance["InstanceId"], instance["State"]["Name"])


def execute(role, command):
    state = read_state()
    response = aws("ssm", "send-command", "--instance-ids", state["instances"][role],
                   "--document-name", "AWS-RunShellScript", "--parameters",
                   json.dumps({"commands": [command], "executionTimeout": ["1800"]}),
                   "--output-s3-bucket-name", state["bucket"], "--output-s3-key-prefix", "commands")
    command_id = response["Command"]["CommandId"]
    state["last_command"] = {"role": role, "id": command_id}
    save(state)
    print(command_id)
    return command_id


def result(command_id, role):
    state = read_state()
    role = role or state["last_command"]["role"]
    command_id = command_id or state["last_command"]["id"]
    output = aws("ssm", "get-command-invocation", "--command-id", command_id,
                 "--instance-id", state["instances"][role])
    print(output["Status"])
    print(output.get("StandardOutputContent", ""))
    print(output.get("StandardErrorContent", ""))


def sync(role):
    state = read_state()
    archive = STATE.parent / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name in ("pyproject.toml", "python", "example", "docs", "README.md", "LICENSE"):
            output.add(ROOT / name, arcname=name, filter=lambda info: None if
                       "__pycache__" in info.name or ".egg-info" in info.name else info)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    key = f"source/{digest}.tar.gz"
    aws("s3api", "put-object", "--bucket", state["bucket"], "--key", key, "--body", str(archive))
    command_id = execute(role, f"set -eu\naws s3 cp {shlex.quote('s3://' + state['bucket'] + '/' + key)} /tmp/gpu2tensor.tar.gz\n"
            f"echo '{digest}  /tmp/gpu2tensor.tar.gz' | sha256sum -c -\n"
            "mkdir -p /opt/gpu2tensor/source\ntar -xzf /tmp/gpu2tensor.tar.gz -C /opt/gpu2tensor/source")
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        time.sleep(2)
        invocation = aws("ssm", "get-command-invocation", "--command-id", command_id,
                         "--instance-id", state["instances"][role])
        if invocation["Status"] == "Success":
            print("Source synchronized.")
            return
        if invocation["Status"] not in {"Pending", "InProgress", "Delayed"}:
            raise RuntimeError(f"Source sync failed: {invocation}")
    raise RuntimeError(f"Source sync still pending; inspect command {command_id} before running code.")


def start(role, port=8000, core=0):
    """Start the worker in the qualified image's existing SDK environment."""
    state = read_state()
    if not 1 <= port <= 65535 or core < 0:
        raise ValueError("Choose a valid port and nonnegative device/core index.")
    unit = "gpu2tensor-worker" if port == 8000 else f"gpu2tensor-worker-{port}"
    environment = "/opt/pytorch" if role == "cuda" else "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0"
    python = environment + "/bin/python"
    command = ["systemd-run", "--collect", f"--unit={unit}",
               "--property=User=ubuntu", "--property=WorkingDirectory=/opt/gpu2tensor/source",
               f"--setenv=GPU2TENSOR_AMI={state['images'][role]['ImageId']}",
               f"--setenv=PATH={environment}/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin"]
    if role == "trainium":
        command += [f"--setenv=NEURON_RT_VISIBLE_CORES={core}",
                    "--setenv=NEURON_PLATFORM_TARGET_OVERRIDE=trn1"]
    else:
        command += [f"--setenv=CUDA_VISIBLE_DEVICES={core}"]
    command += [python, "-m", "gpu2tensor.worker", "--backend", role, "--port", str(port)]
    execute(role, "set -eu\n" + shlex.join([python, "-m", "pip", "install", "--no-deps", "-e", "/opt/gpu2tensor/source"]) +
            f"\nsystemctl stop {unit}.service || true\n" + shlex.join(command))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("status")
    tunnel = commands.add_parser("tunnel")
    tunnel.add_argument("role", choices=["cuda", "trainium"])
    tunnel.add_argument("--port", type=int, default=8000)
    tunnel.add_argument("--remote-port", type=int, default=8000)
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("role", choices=["cuda", "trainium"])
    launch_parser.add_argument("--zone", default="us-east-1f", help="Availability zone, or auto for regional placement")
    launch_parser.add_argument("--type", dest="instance_type", help="CUDA: g4dn.xlarge, g5/g6 xlarge or 2xlarge; Trainium: trn1.2xlarge")
    for name in ("exec", "sync", "start", "terminate"):
        sub = commands.add_parser(name)
        sub.add_argument("role", choices=["cuda", "trainium"])
        if name == "exec":
            sub.add_argument("shell_command")
        if name == "start":
            sub.add_argument("--port", type=int, default=8000)
            sub.add_argument("--core", type=int, default=0)
    results = commands.add_parser("result")
    results.add_argument("id", nargs="?")
    results.add_argument("--role", choices=["cuda", "trainium"])
    args = parser.parse_args()
    if args.command == "init": initialize()
    elif args.command == "status": status()
    elif args.command == "launch": launch(args.role, args.zone, args.instance_type)
    elif args.command == "exec": execute(args.role, args.shell_command)
    elif args.command == "sync": sync(args.role)
    elif args.command == "start": start(args.role, args.port, args.core)
    elif args.command == "result": result(args.id, args.role)
    elif args.command == "tunnel":
        state = read_state()
        environment = os.environ.copy()
        environment["PATH"] = str(ROOT / ".local/tools") + os.pathsep + environment.get("PATH", "")
        subprocess.run(["aws", "ssm", "start-session", "--region", REGION,
                        "--target", state["instances"][args.role],
                        "--document-name", "AWS-StartPortForwardingSession",
                        "--parameters", json.dumps({"portNumber": [str(args.remote_port)], "localPortNumber": [str(args.port)]})],
                       env=environment, check=True)
    elif args.command == "terminate":
        state = read_state()
        print(aws("ec2", "terminate-instances", "--instance-ids", state["instances"][args.role]))


if __name__ == "__main__":
    main()
