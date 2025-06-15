import os
import time
import json
import requests
from pathlib import Path
from socket import create_connection

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

        # === STEP 4: Upload your project to pod
        print("📤 Uploading local folder to pod...")
        os.system(f"scp -r -i {ssh_key_path} -P {ssh_port} ./llm-builder/ {ssh_user}@{pod_ip}:/workspace/")

        # === STEP 5: Trigger training remotely
        print("🚀 Launching training on pod...")
        print("🪵 Live training logs:")
        # Upload the unified training script
        print("📤 Uploading training script...")
        os.system(f"scp -i {ssh_key_path} -P {ssh_port} ./runpod_train.sh {ssh_user}@{pod_ip}:/workspace/runpod_train.sh")
        
        # Set environment variables and execute the script
        env_vars = f"HF_TOKEN={config.HF_TOKEN}" if hasattr(config, 'HF_TOKEN') and config.HF_TOKEN else ""
        ssh_cmd = f'ssh -i {ssh_key_path} -p {ssh_port} {ssh_user}@{pod_ip} "{env_vars} chmod +x /workspace/runpod_train.sh && {env_vars} /workspace/runpod_train.sh"'
        with os.popen(ssh_cmd) as stream:
            for line in stream:
                print(line, end="")
        
        # Download the trained model
        print("📥 Downloading trained model...")
        os.system(f"scp -r -i {ssh_key_path} -P {ssh_port} {ssh_user}@{pod_ip}:/workspace/output ./")
        print("✅ Model downloaded to ./output/")
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