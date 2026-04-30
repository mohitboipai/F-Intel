import subprocess
import socket
import re
import time
import sys
import threading
import os

PORT = 8082

def is_server_running():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', PORT)) == 0

def ensure_dataserver():
    if is_server_running():
        return
    
    print("Starting DataServer...")
    # Start in background
    subprocess.Popen([sys.executable, "DataServer.py"], 
                     stdout=subprocess.DEVNULL, 
                     stderr=subprocess.DEVNULL)
    
    # Wait up to 15s
    for _ in range(15):
        if is_server_running():
            return
        time.sleep(1)
        
    print("Error: DataServer failed to start within 15 seconds.")
    sys.exit(1)

def check_cloudflared():
    try:
        subprocess.run(["cloudflared", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("cloudflared is not installed. Install it:")
        print("Windows: download from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
        print("Mac:     brew install cloudflared")
        print("Linux:   sudo apt install cloudflared")
        print("           OR download binary from the URL above")
        print("Then run this script again.")
        sys.exit(1)

def heartbeat():
    while True:
        time.sleep(300)
        print("Tunnel active")

def main():
    ensure_dataserver()
    check_cloudflared()
    
    print("Starting Cloudflare Tunnel...")
    # cloudflared tunnel logs to stderr by default
    proc = subprocess.Popen(["cloudflared", "tunnel", "--url", f"http://localhost:{PORT}"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    
    url_found = False
    
    try:
        for line in proc.stdout:
            if not url_found and "https://" in line and "trycloudflare.com" in line:
                match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                if match:
                    url = match.group(0)
                    print("\n============================================")
                    print("F-Intel is ready on your phone\n")
                    print(f"Open this in your phone browser:")
                    print(f"{url}\n")
                    print("Tip: bookmark it or add to home screen")
                    print("============================================\n")
                    
                    with open(".tunnel_url", "w") as f:
                        f.write(url)
                    url_found = True
                    
                    # Start heartbeat
                    threading.Thread(target=heartbeat, daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nTunnel stopped.")
        proc.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
