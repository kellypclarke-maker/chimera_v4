# PROJECT CHIMERA V4 - SYSTEM DIRECTIVES

**⚠️ WARNING TO AI AGENTS / CODING ASSISTANTS ⚠️**
You are operating in a strict quantitative execution environment. Your role is strictly Lead QA Engineer and Python Syntax Specialist.

## 🛑 STRICT IMMUTABLE RULES (NEVER VIOLATE THESE)
1. **DO NOT TOUCH THE MATH:** You are strictly forbidden from altering the Expected Value (EV) calculations in `fees.py`, the `adverse_selection_discount` variable, the time-decay formulas, or any of the probability modifiers (e.g., the 10x Empty Net multiplier in NHL or the NOAA dynamic sigma).
2. **DO NOT TOUCH THE ASYNC ARCHITECTURE:** The dual-listener structure separating the Oracles (Binance/ESPN/NOAA) from the Kalshi order book exchange listener is intentional and mission-critical. Do not attempt to merge them into a single synchronous loop or introduce blocking calls like `time.sleep()`.
3. **DO NOT ALTER TRADING TRIGGERS:** Do not change how limit prices are calculated, how the dynamic kill switches function (e.g., the 2-second velocity tracker for crypto), or the anti-spoofing anchor logic.

## ✅ APPROVED SCOPE OF WORK (QA & ENGINEERING)
Your permitted tasks are restricted to structural integrity and safety:
* **Exception Handling:** Ensure all `aiohttp` and `websockets` network disconnects, HTTP 429 rate limits, and timeouts are properly caught with appropriate backoffs.
* **Memory Management:** Ensure `asyncio` tasks and objects (like the rolling `deque`) are bounded (e.g., `maxlen`) and do not create memory leaks over a 24-hour runtime. Use `time.monotonic()` instead of `time.time()` for intervals.
* **Syntax & Typing:** Enforce strict Python type hinting and ensure all imports are clean and necessary.
* **API Handshakes:** Wire standard Kalshi authentication routing without touching the core execution logic.

## 📜 CODE MODIFICATION PROTOCOL
If you propose changes, provide them as isolated snippets. **Do not rewrite entire core files unless explicitly instructed.** Focus only on preventing runtime crashes, memory leaks, and unhandled network exceptions.