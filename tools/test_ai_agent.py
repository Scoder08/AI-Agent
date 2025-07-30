import sys
import os
import asyncio

# Add the project root to sys.path so 'utils' becomes importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ai_session import Session

new_session = Session('testsession', 'shresth', exp=1234567890)

async def main():
    while True:
        a = input("Enter your query: ")
        output = new_session.run_query(a, sync=True)
        response = ""
        async for chunk in output:
            response += chunk
        print(f"Response : {response}")
        if a.lower() == 'exit':
            break

if __name__ == "__main__":
    asyncio.run(main())
