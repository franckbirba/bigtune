import os
import time
import json
import requests
import yaml
import shutil
from pathlib import Path
from socket import create_connection

# Get the directory where this script is located (BigTune installation)
SCRIPT_DIR = Path(__file__).parent

# Import configuration
try:
    from bigtune.config import config
    API_KEY = config.RUNPOD_API_KEY
    IMAGE_NAME = config.RUNPOD_IMAGE_NAME
    VOLUME_NAME = config.VOLUME_NAME
    DATASET_DIR = config.DATASET_DIR
    CONFIG_FILE = config.CONFIG_FILE
    OUTPUT_DIR = config.OUTPUT_DIR
    GPU_TYPE = config.GPU_TYPE
    MACHINE_SIZE = config.MACHINE_SIZE
    SSH_KEY_PATH = config.SSH_KEY_PATH
    VOLUME_SIZE_GB = config.VOLUME_SIZE_GB
    CONTAINER_DISK_SIZE_GB = config.CONTAINER_DISK_SIZE_GB
    MIN_VCPU_COUNT = config.MIN_VCPU_COUNT
    MIN_MEMORY_GB = config.MIN_MEMORY_GB
except ImportError:
    # Fallback to hardcoded values if bigtune package not available
    print("⚠️  Using fallback configuration - install bigtune package for .env support")
    API_KEY = os.getenv('RUNPOD_API_KEY', '')
    IMAGE_NAME = "nvidia/cuda:12.1.1-devel-ubuntu22.04"
    VOLUME_NAME = "llm-builder"
    DATASET_DIR = "./datasets"
    CONFIG_FILE = "config/positivity-lora.yaml"
    OUTPUT_DIR = "./output"
    GPU_TYPE = "NVIDIA A40"
    MACHINE_SIZE = "petite"
    SSH_KEY_PATH = os.path.expanduser("~/.ssh/runpod_rsa")
    VOLUME_SIZE_GB = 50
    CONTAINER_DISK_SIZE_GB = 50
    MIN_VCPU_COUNT = 2
    MIN_MEMORY_GB = 15

def parse_and_upload_external_datasets(config_file_path, ssh_key_path, ssh_port, ssh_user, pod_ip):
    """Parse dataset paths from external config and upload them to RunPod"""
    try:
        # Parse the config file to extract dataset paths
        with open(config_file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        datasets = config_data.get('datasets', [])
        if not datasets:
            print("⚠️ No datasets found in config file")
            return True
        
        print(f"📂 Found {len(datasets)} datasets to upload")
        
        # Create datasets directory on RunPod
        mkdir_cmd = f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {ssh_key_path} -p {ssh_port} {ssh_user}@{pod_ip} "mkdir -p /workspace/datasets"'
        result = os.system(mkdir_cmd)
        if result != 0:
            print(f"⚠️ Failed to create datasets directory")
            return False
        
        # Upload each dataset file
        uploaded_files = []
        for dataset in datasets:
            dataset_path = Path(dataset.get('path', ''))
            if dataset_path.exists():
                filename = dataset_path.name
                upload_cmd = f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {ssh_key_path} -P {ssh_port} {dataset_path} {ssh_user}@{pod_ip}:/workspace/datasets/"
                print(f"📤 Uploading dataset: {filename}")
                result = os.system(upload_cmd)
                if result == 0:
                    uploaded_files.append(filename)
                    print(f"✅ Uploaded: {filename}")
                else:
                    print(f"❌ Failed to upload: {filename}")
                    return False
            else:
                print(f"❌ Dataset file not found: {dataset_path}")
                return False
        
        print(f"✅ Successfully uploaded {len(uploaded_files)} dataset files")
        return True
        
    except Exception as e:
        print(f"❌ Error parsing/uploading datasets: {e}")
        return False

def create_runpod_compatible_config(config_file_path, output_path):
    """Create a RunPod-compatible version of the config with relative paths"""
    try:
        with open(config_file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update dataset paths to use RunPod workspace structure
        if 'datasets' in config_data:
            for dataset in config_data['datasets']:
                if 'path' in dataset:
                    original_path = Path(dataset['path'])
                    dataset['path'] = f"datasets/{original_path.name}"
        
        # Update other paths to use workspace structure
        if 'dataset_prepared_path' in config_data:
            config_data['dataset_prepared_path'] = "output/prepared"
        
        if 'output_dir' in config_data:
            # Extract just the final directory name for output
            original_output = Path(config_data['output_dir'])
            config_data['output_dir'] = f"output/{original_output.name}"
        
        # Write the modified config
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        print(f"✅ Created RunPod-compatible config: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating RunPod config: {e}")
        return False

def get_download_destination(config_file_path):
    """Determine the correct download destination based on config type"""
    try:
        # If using external config, parse the original output_dir
        if Path(config_file_path).is_absolute():
            with open(config_file_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            external_output_dir = config_data.get('output_dir', '')
            if external_output_dir and Path(external_output_dir).is_absolute():
                # Get the parent directory of the external output (where we want to download)
                # This will be: /Users/franckbirba/DEV/TEST-CREWAI/bigacademy/ 
                # So when scp adds /output, it becomes: /Users/franckbirba/DEV/TEST-CREWAI/bigacademy/output/
                external_parent = Path(external_output_dir).parent.parent
                print(f"📂 Using external output destination: {external_parent}/")
                return str(external_parent) + "/"
        
        # Default to BigTune's local directory
        print(f"📂 Using BigTune's local output destination: ./")
        return "./"
        
    except Exception as e:
        print(f"⚠️ Error determining download destination: {e}")
        print(f"📂 Falling back to BigTune's local output destination: ./")
        return "./"

def list_available_gpus():
    url = "https://api.runpod.io/graphql"
    headers = {"Authorization": API_KEY}
    query = """
    query {
      gpuTypes(input: {}) {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        maxGpuCountSecureCloud
        maxGpuCountCommunityCloud
      }
    }
    """
    resp = requests.post(url, json={"query": query}, headers=headers)
    res = resp.json().get("data", {}).get("gpuTypes", [])

    return [
        g for g in res if (
            (g["secureCloud"] and g["maxGpuCountSecureCloud"] > 0) or
            (g["communityCloud"] and g["maxGpuCountCommunityCloud"] > 0)
        )
    ]

# === STEP 1: CREATE A POD ===
# === STEP 1: CREATE A POD (PATCHED) ===
def create_pod():
    gpu_priority = {
        "petite": [
            "A40",
            "RTX A2000", "RTX 3070", "RTX 3080", "RTX 3080 Ti", "RTX 3090",
            "RTX 3090 Ti", "RTX 4070 Ti", "RTX 4080", "RTX 4080 SUPER", "RTX 5080",
            "RTX 2000 Ada", "RTX 4000 Ada", "RTX A4000", "RTX A4500", "RTX A5000",
            "V100 FHHL", "Tesla V100", "V100 SXM2", "V100 SXM2 32GB"
        ],
        "moyenne": ["NVIDIA RTX A4000", "NVIDIA GeForce RTX 4090"],
        "grosse": ["NVIDIA RTX A6000", "NVIDIA A100"]
    }

    available = list_available_gpus()
    print("🔍 GPU disponibles :", ", ".join([name['displayName'] for name in available]))
    priority_list = gpu_priority.get(MACHINE_SIZE, [])
    chosen_gpu = None

    for pref in priority_list:
        match = next((g for g in available if pref in g["displayName"]), None)
        if match:
            chosen_gpu = match["id"]
            print(f"✅ GPU sélectionné : {match['displayName']} ({chosen_gpu})")
            break

    if not chosen_gpu:
        print(f"⚠️ Aucun GPU prioritaire trouvé pour MACHINE_SIZE='{MACHINE_SIZE}'. Tentative avec un GPU alternatif de même catégorie...")

        # Définir les contraintes mémoire max pour chaque taille
        max_mem_map = {
            "petite": 20,
            "moyenne": 48,
            "grosse": 9999  # no upper bound
        }
        mem_limit = max_mem_map.get(MACHINE_SIZE, 20)

        # Trouver un fallback raisonnable dans la même gamme
        fallback = next((g for g in available if g["memoryInGb"] <= mem_limit), None)
        if fallback:
            chosen_gpu = fallback["id"]
            print(f"✅ Fallback GPU sélectionné : {fallback['displayName']} ({chosen_gpu})")
        else:
            raise SystemExit(f"❌ Aucun GPU adapté trouvé sous {mem_limit}GB pour MACHINE_SIZE='{MACHINE_SIZE}'.")

    with open(SSH_KEY_PATH + ".pub", "r") as f:
        ssh_pub_key = f.read().strip()

    query = f"""
    mutation {{
        podFindAndDeployOnDemand(input: {{
            cloudType: SECURE,
            gpuCount: 1,
            volumeInGb: {VOLUME_SIZE_GB},
            containerDiskInGb: {CONTAINER_DISK_SIZE_GB},
            minVcpuCount: {MIN_VCPU_COUNT},
            minMemoryInGb: {MIN_MEMORY_GB},
            gpuTypeId: "{chosen_gpu}",
            name: "llm-builder-job",
            imageName: "{IMAGE_NAME}",
            ports: "22/tcp",
            dockerArgs: "bash -c 'apt update;DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y;mkdir -p ~/.ssh;cd ~/.ssh;chmod 700 ~/.ssh;echo \\\"{ssh_pub_key}\\\" >> authorized_keys;chmod 700 authorized_keys;service ssh start;sleep infinity'",
            volumeMountPath: "/workspace"
        }}) {{
            id
            imageName
            machineId
        }}
    }}
    """

    url = "https://api.runpod.io/graphql"
    headers = {"Authorization": API_KEY}

    resp = requests.post(url, json={"query": query}, headers=headers)
    data = resp.json()
    if "errors" in data or data.get("data", {}).get("podFindAndDeployOnDemand") is None:
        print("❌ Échec de création du pod avec GPU :", chosen_gpu)
        # Nouvelle tentative avec autres GPU compatibles
        # Définir les contraintes mémoire max pour chaque taille
        max_mem_map = {
            "petite": 20,
            "moyenne": 48,
            "grosse": 9999  # no upper bound
        }
        mem_limit = max_mem_map.get(MACHINE_SIZE, 20)
        alt_gpus = sorted(
            [g for g in available if g["memoryInGb"] <= mem_limit and g["id"] != chosen_gpu],
            key=lambda g: g["memoryInGb"]
        )
        for g in alt_gpus:
            print(f"🔄 Tentative avec {g['displayName']} ({g['id']})...")
            print(g)
            new_query = query.replace(f'gpuTypeId: "{chosen_gpu}"', f'gpuTypeId: "{g["id"]}"')
            new_resp = requests.post(url, json={"query": new_query}, headers=headers)
            new_data = new_resp.json()
            if "errors" not in new_data and new_data.get("data", {}).get("podFindAndDeployOnDemand"):
                print(f"✅ Nouveau GPU sélectionné : {g['displayName']}")
                return new_data
            else :
                print(new_data)
            print("❌ Toujours pas de pod dispo.")
        raise SystemExit("🚫 Aucun GPU disponible après plusieurs tentatives.")
    return data

# === STEP 2: MONITOR POD STATUS ===
def wait_for_pod_ready(pod_id):
    url = "https://api.runpod.io/graphql"
    headers = {"Authorization": API_KEY}
    while True:
        query = f'''
        query {{
          pod(input: {{ podId: "{pod_id}" }}) {{
            id
            runtime {{
              ports {{
                ip
                privatePort
                publicPort
                type
              }}
            }}
          }}
        }}
        '''
        response = requests.post(url, json={"query": query}, headers=headers)
        response_json = response.json()

        if "data" not in response_json or response_json["data"] is None or "pod" not in response_json["data"]:
            print("❌ Erreur : impossible de récupérer les infos du pod.")
            print("Réponse complète :", json.dumps(response_json, indent=2))
            raise SystemExit("⛔ Arrêt : données pod indisponibles.")

        pod = response_json["data"]["pod"]
        runtime = pod.get("runtime", {}) or {}
        ports = runtime.get("ports", []) or []
        
        print(f"🔍 Full pod status:")
        print(f"   Pod: {json.dumps(pod, indent=2)}")
        print(f"🔍 Checking ports: {ports}")
        
        for port in ports:
            print(f"   Port details: {port}")
            if (
                port.get("type") == "tcp"
                and str(port.get("privatePort")) == "22"
            ):
                print(f"   ✅ SSH port 22 found, public port: {port.get('publicPort')}")
                return port.get("ip"), port.get("publicPort")
        print("   ⏳ No SSH port 22 found yet, waiting...")
        time.sleep(10)

def wait_ssh_ready(ip, port=22, timeout=180):
    print(f"⏳ Waiting for SSH to become available at {ip}:{port}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with create_connection((ip, port), timeout=5):
                print("✅ SSH is ready!")
                return True
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"⛔ SSH not ready after {timeout} seconds.")

# === MAIN LOGIC ===
def main():
    # Validate API key
    if not API_KEY:
        print("❌ Error: RUNPOD_API_KEY not found!")
        print("Please set your RunPod API key in .env file or environment variable")
        exit(1)
    
    result = create_pod()
    pod_info = result["data"]["podFindAndDeployOnDemand"]
    pod_id = pod_info["id"]
    print(f"Created pod: {pod_id}")
    try:
        pod_ip, ssh_port = wait_for_pod_ready(pod_id)
        print(f"✅ Pod is ready! IP: {pod_ip}, SSH Port: {ssh_port}")
        wait_ssh_ready(pod_ip, ssh_port)

        ssh_user = "root"
        ssh_key_path = SSH_KEY_PATH
        
        # Verify SSH key exists
        if not os.path.exists(ssh_key_path):
            print(f"❌ SSH key not found: {ssh_key_path}")
            return None
        
        print(f"🔑 Using SSH key: {ssh_key_path}")

        # === STEP 4: Upload your project to pod
        print("📤 Uploading local folder to pod...")
        
        # Determine upload strategy based on config file location
        using_external_config = Path(config.CONFIG_FILE).is_absolute()
        
        if using_external_config:
            # When using external config, upload from external project
            config_dir = Path(config.CONFIG_FILE).parent.parent
            external_llm_builder = config_dir / "llm-builder"
            
            if external_llm_builder.exists():
                print(f"📂 Using external llm-builder from: {external_llm_builder}")
                upload_cmd = f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -i {ssh_key_path} -P {ssh_port} {external_llm_builder}/ {ssh_user}@{pod_ip}:/workspace/"
                print(f"🔧 Upload command: {upload_cmd}")
                result = os.system(upload_cmd)
                if result != 0:
                    print(f"⚠️ Upload failed with exit code: {result}")
                    return None
            else:
                # Smart external config and dataset upload
                print(f"📂 Using smart external config and dataset upload")
                
                # First create necessary directories
                mkdir_cmd = f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {ssh_key_path} -p {ssh_port} {ssh_user}@{pod_ip} "rm -rf /workspace/config /workspace/datasets /workspace/output && mkdir -p /workspace/config /workspace/datasets /workspace/output"'
                print(f"🔧 Creating clean directories")
                result = os.system(mkdir_cmd)
                if result != 0:
                    print(f"⚠️ Failed to create directories with exit code: {result}")
                    return None
                
                # Create RunPod-compatible config
                config_file_path = Path(config.CONFIG_FILE)
                temp_config_path = "/tmp/runpod_config.yaml"
                if not create_runpod_compatible_config(config_file_path, temp_config_path):
                    return None
                
                # Upload the RunPod-compatible config
                config_upload_cmd = f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {ssh_key_path} -P {ssh_port} {temp_config_path} {ssh_user}@{pod_ip}:/workspace/config/{config_file_path.name}"
                print(f"🔧 Uploading RunPod-compatible config")
                result = os.system(config_upload_cmd)
                if result != 0:
                    print(f"⚠️ Config upload failed with exit code: {result}")
                    return None
                else:
                    print("✅ RunPod-compatible config uploaded successfully")
                
                # Upload datasets by parsing config
                if not parse_and_upload_external_datasets(config_file_path, ssh_key_path, ssh_port, ssh_user, pod_ip):
                    return None
                
                # Clean up temp config
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
        else:
            # When using BigTune's internal config, use BigTune's llm-builder
            llm_builder_paths = [
                Path("./llm-builder"),  # Current working directory  
                SCRIPT_DIR / "llm-builder"  # BigTune installation
            ]
            
            llm_builder_path = None
            for path in llm_builder_paths:
                if path.exists():
                    llm_builder_path = path
                    break
            
            if not llm_builder_path:
                print("❌ Error: Could not find llm-builder directory")
                return None
                
            print(f"📂 Using BigTune's llm-builder from: {llm_builder_path}")
            upload_cmd = f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -i {ssh_key_path} -P {ssh_port} {llm_builder_path}/ {ssh_user}@{pod_ip}:/workspace/"
            print(f"🔧 Upload command: {upload_cmd}")
            result = os.system(upload_cmd)
            if result != 0:
                print(f"⚠️ Upload failed with exit code: {result}")
                return None

        # === STEP 5: Trigger training remotely
        print("🚀 Launching training on pod...")
        print("🪵 Live training logs:")
        # Upload the unified training script
        print("📤 Uploading training script...")
        runpod_script_path = SCRIPT_DIR / "runpod_train.sh"
        script_upload_cmd = f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {ssh_key_path} -P {ssh_port} {runpod_script_path} {ssh_user}@{pod_ip}:/workspace/runpod_train.sh"
        print(f"🔧 Script upload command: {script_upload_cmd}")
        result = os.system(script_upload_cmd)
        if result != 0:
            print(f"⚠️ Script upload failed with exit code: {result}")
            return None
        
        # Set environment variables and execute the script
        env_vars = []
        if hasattr(config, 'HF_TOKEN') and config.HF_TOKEN:
            env_vars.append(f"HF_TOKEN={config.HF_TOKEN}")
        if hasattr(config, 'CONFIG_FILE') and config.CONFIG_FILE:
            env_vars.append(f"CONFIG_FILE={config.CONFIG_FILE}")
        
        env_str = " ".join(env_vars) + " " if env_vars else ""
        ssh_cmd = f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {ssh_key_path} -p {ssh_port} {ssh_user}@{pod_ip} "{env_str}chmod +x /workspace/runpod_train.sh && {env_str}/workspace/runpod_train.sh"'
        print(f"🔧 SSH training command: {ssh_cmd}")
        
        # Execute training with proper error handling and real-time output
        import subprocess
        try:
            print("🚀 Starting training execution...")
            process = subprocess.Popen(
                ssh_cmd, 
                shell=True, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line, end="")
            
            # Wait for completion and get return code
            return_code = process.wait()
            
            if return_code != 0:
                print(f"❌ Training command failed with exit code: {return_code}")
                return None
            else:
                print("✅ Training command completed successfully")
                
        except Exception as e:
            print(f"❌ SSH execution failed: {e}")
            return None
        
        # Test SSH connection before download
        print("🔗 Testing SSH connection...")
        test_ssh_cmd = f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -i {ssh_key_path} -p {ssh_port} {ssh_user}@{pod_ip} "echo SSH connection test successful"'
        print(f"🔧 SSH test command: {test_ssh_cmd}")
        ssh_test_result = os.system(test_ssh_cmd)
        
        if ssh_test_result != 0:
            print("❌ SSH connection test failed - pod may be unreachable")
            return None
        else:
            print("✅ SSH connection test successful")
        
        # Check if output directory exists on pod before downloading
        print("🔍 Checking for training output on pod...")
        check_cmd = f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {ssh_key_path} -p {ssh_port} {ssh_user}@{pod_ip} "ls -la /workspace/output"'
        print(f"🔧 Check command: {check_cmd}")
        check_result = os.system(check_cmd)
        
        if check_result != 0:
            print("⚠️ No output directory found on pod - training may have failed")
        else:
            print("✅ Output directory found on pod")
        
        # Download the trained model
        print("📥 Downloading trained model...")
        
        # Determine correct download destination
        download_dest = get_download_destination(config.CONFIG_FILE)
        
        # Ensure download destination exists
        Path(download_dest).mkdir(parents=True, exist_ok=True)
        
        download_cmd = f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r -i {ssh_key_path} -P {ssh_port} {ssh_user}@{pod_ip}:/workspace/output {download_dest}"
        print(f"🔧 Download command: {download_cmd}")
        
        # Add retry logic for download
        max_retries = 3
        for attempt in range(max_retries):
            print(f"📥 Download attempt {attempt + 1}/{max_retries}...")
            result = os.system(download_cmd)
            if result == 0:
                print(f"✅ Model downloaded to {download_dest}output/")
                break
            else:
                print(f"⚠️ Download attempt {attempt + 1} failed with exit code: {result}")
                if attempt < max_retries - 1:
                    print("⏳ Waiting 5 seconds before retry...")
                    import time
                    time.sleep(5)
                else:
                    print("❌ All download attempts failed")
                    return None
    except Exception as e:
        print("❌ Une erreur est survenue :", str(e))
    finally:
        print("🧹 Cleaning up: deleting pod...")
        delete_query = f'''
        mutation {{
            podTerminate(input: {{ podId: "{pod_id}" }})
        }}
        '''
        headers = {"Authorization": API_KEY}
        url = "https://api.runpod.io/graphql"
        response = requests.post(url, json={"query": delete_query}, headers=headers)
        result = response.json()
        
        if "errors" in result:
            print("❌ Failed to delete pod:")
            print(json.dumps(result, indent=2))
            print(f"💡 Manual cleanup needed - Pod ID: {pod_id}")
        else:
            print("✅ Pod deleted successfully.")

if __name__ == "__main__":
    main()