import subprocess
import time
import os
import sys
import webbrowser
import signal

def main():
    print("🚀 Initializing GoTrainer System (Multi-Process)...")
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SERVER_SCRIPT = os.path.join(BASE_DIR, "web", "backend", "server.py")
    TRAINER_SCRIPT = os.path.join(BASE_DIR, "learning", "train_worker.py")
    
    # 1. Start Training Process
    print("   - Launching Training Worker...")
    # Redirect output to file for debugging
    worker_log = open(os.path.join(BASE_DIR, "worker.log"), "w", encoding="utf-8")
    trainer_process = subprocess.Popen(
        [sys.executable, "-u", TRAINER_SCRIPT],
        cwd=BASE_DIR,
        stdout=worker_log,
        stderr=subprocess.STDOUT
    )

    # 2. Start Web Server
    print("   - Launching Web Server...")
    # uvicorn command
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "web.backend.server:app", "--host", "127.0.0.1", "--port", "8050"],
        cwd=BASE_DIR,
        # creationflags=subprocess.CREATE_NEW_CONSOLE 
    )
    
    print("✅ System Started!")
    print("   - Web Interface: http://127.0.0.1:8050")
    
    # Open browser
    time.sleep(2)
    try:
        webbrowser.open("http://127.0.0.1:8050")
    except:
        pass
    
    print("⏳ Monitoring processes... (Press CTRL+C to stop)")
    try:
        while True:
            time.sleep(1)
            # Check if processes are alive
            if trainer_process.poll() is not None:
                print("⚠️ Training Worker exited unexpectedly.")
                print("--- Worker Log (Last 20 lines) ---")
                try:
                    # Give OS a moment to release locks
                    time.sleep(0.5)
                    with open("worker.log", "r", encoding="utf-8", errors='replace') as f:
                        lines = f.readlines()
                        for line in lines[-20:]:
                            print(line.strip())
                except Exception as e:
                    print(f"Could not read worker.log: {e}")
                break
            if server_process.poll() is not None:
                print("⚠️ Web Server exited unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
    finally:
        # Ensure cleanup happens on any exit
        print("🧹 Cleaning up processes...")
        try:
            if trainer_process.poll() is None:
                trainer_process.terminate()
        except:
            pass
        try:
            if server_process.poll() is None:
                server_process.terminate()
        except:
            pass
        try:
            worker_log.close()
        except:
            pass
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
