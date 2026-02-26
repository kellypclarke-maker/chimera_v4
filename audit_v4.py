import os
from pathlib import Path

def audit():
    print("\n🚀 Project Chimera v4: Credential Audit")
    print("="*40)
    
    # 1. Check Env Vars
    vars_to_check = ["KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PATH"]
    for var in vars_to_check:
        val = os.getenv(var)
        if val:
            print(f"✅ {var}: LOADED")
            # 2. Check File Path if it's the Private Key
            if var == "KALSHI_PRIVATE_KEY_PATH":
                path = Path(val)
                if path.exists():
                    print(f"   ∟ 📂 RSA Key found at: {path}")
                else:
                    print(f"   ∟ ❌ ERROR: RSA file NOT found at {path}")
                    print(f"      (Note: WSL needs paths like /mnt/c/Users/...)")
        else:
            print(f"❌ {var}: MISSING from environment")

    if os.getenv("BINANCE_API_KEY"):
        print("✅ BINANCE_API_KEY: LOADED")

if __name__ == "__main__":
    audit()
