import asyncio
import json
import zmq
import zmq.asyncio
import os
import traceback
import time
from cerebras.cloud.sdk import AsyncCerebras
from utilities.database_functions import load_json, update_memory, build_system_prompt, wrap_tools_for_groq

client = AsyncCerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY")
)
context = zmq.asyncio.Context()

zmq_req_socket = context.socket(zmq.REQ)
zmq_req_socket.connect("tcp://localhost:5555")

zmq_sub_socket = context.socket(zmq.SUB)
zmq_sub_socket.connect("tcp://localhost:5556")
zmq_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
zmq_req_lock = asyncio.Lock()

AI_MODEL = "gpt-oss-120b"
IDENTITY_PATH = "database/identity.json"
MEMORY_PATH = "database/memory.json"
TOOLS_PATHS = [
    "tools/database_tools.json", 
    "tools/navigate_tools.json", 
    "tools/thinking_tools.json"
]


async def background_status_monitor(message_queue):
    print("[System: Background Monitor Listening for ROS 2 feedback...]")
    while True:
        try:
            message = await zmq_sub_socket.recv_json()
            if "status" in message:
                alert = f"[SYSTEM NOTIFICATION]: {message.get('event')} - {message['status']}"
                await message_queue.put(alert)
                print(f"\n🔔 {alert}") 
        except Exception as e:
            print(f"[System Error in Monitor]: {e}")
            await asyncio.sleep(1) 

async def get_async_input(prompt: str):
    return await asyncio.to_thread(input, prompt)

async def handle_tool_calls(tool_call):
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

    if name == "update_memory":
        print(f"[System: Saving to disk -> {arguments.get('category')}: {arguments.get('new_info')}]")
        result = update_memory(MEMORY_PATH, arguments.get("category"), arguments.get("new_info"))
        return str(result)

    elif name in ["blind_move", "navigate_to_pose", "stop_motors", "get_vision", "find_object"]:
        payload = {"command": name, **arguments}
        
        print(f"[System: Dispatching {name} command to ROS 2...]")
        print(f"Payload: {payload}")
        async with zmq_req_lock:
            await zmq_req_socket.send_json(payload)
            feedback = await zmq_req_socket.recv_json()
        print(f"[System: ROS 2 Acknowledged: {feedback.get('status', 'Unknown')}]")
        
        return json.dumps(feedback)
    
    return "Error: Unknown tool."

async def robot_brain(message_queue, messages, tools_file):
    while True:
        user_input = await message_queue.get()
        
        if user_input.lower() in ['exit', 'quit']:
            print("🤖 DJ: Powering down. See you later!")
            break

        messages.append({"role": "user", "content": user_input})

        while True:
            try:
                # Dynamically construct arguments so we don't pass 'None' to Groq
                api_args = {
                    "model": AI_MODEL,
                    "messages": messages,
                }
                if tools_file:
                    api_args["tools"] = tools_file
                    api_args["tool_choice"] = "auto"

                # 1. Start the timer
                start_time = time.time()

                response = await asyncio.wait_for(
                        client.chat.completions.create(**api_args, temperature=0.8, max_completion_tokens=1024),
                        timeout=10.0  # Adjust this depending on expected API speed
                    )
                
                # 2. Stop the timer
                end_time = time.time()

                # 3. Calculate and print Tokens per Second (TPS)
                if response.usage and response.usage.completion_tokens:
                    time_taken = end_time - start_time
                    completion_tokens = response.usage.completion_tokens
                    tps = completion_tokens / time_taken if time_taken > 0 else 0
                    print(f"⚡ [System: Brain Speed - {tps:.2f} tokens/sec ({completion_tokens} tokens in {time_taken:.2f}s)]")

                response_message = response.choices[0].message
                
                # Safely serialize the assistant message to append to history
                # Groq requires content to be an empty string rather than None when calling tools
                assistant_message = {
                    "role": "assistant",
                    "content": response_message.content or "",
                }

                if response_message.tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": t.id,
                            "type": t.type,
                            "function": {
                                "name": t.function.name,
                                "arguments": t.function.arguments
                            }
                        } for t in response_message.tool_calls
                    ]
                    messages.append(assistant_message)

                    # Process the tools
                    tool_tasks = []
                    for tool_call in response_message.tool_calls:
                        tool_tasks.append(handle_tool_calls(tool_call))
                    
                    tool_results = await asyncio.gather(*tool_tasks)
                    
                    # Append the results of the tools back to the conversation
                    for tool_call, result in zip(response_message.tool_calls, tool_results):
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": result
                        })
                else:
                    # No tools were called; final response generated
                    messages.append(assistant_message)
                    print(f"\n🤖 DJ: {response_message.content}\n")
                    break
                    
            except TimeoutError:
                    # In Python 3.11+, asyncio.TimeoutError is just TimeoutError
                    print("\n🔴 [Brain Error - Timeout]: Cerebras took too long. It's likely waiting out a 429 Rate Limit. Give it a few seconds!")
                    break 
            except Exception as e:
                    # Catch literal 429 errors just in case they slip past the timeout
                    if "429" in str(e):
                        print("\n🔴 [Rate Limit Hit]: Whoa, DJ's brain is overloaded! Please wait a minute before sending more commands.")
                    else:
                        print(f"\n🔴 [Brain Error - SDK Crashed]: {e}")
                        traceback.print_exc()
                    break
               

async def main():
    identity_file = load_json(IDENTITY_PATH)
    memory_file = load_json(MEMORY_PATH)
    
    tools_file = []
    for path in TOOLS_PATHS:
        raw_tools = load_json(path)
        tools_file.extend(wrap_tools_for_groq(raw_tools))

    system_prompt = build_system_prompt(identity_file, memory_file)
    messages = [{"role": "system", "content": system_prompt}]

    print("🤖 DJ (Dang Junior) is online! (Type 'exit' to quit)\n")

    message_queue = asyncio.Queue()

    monitor_task = asyncio.create_task(background_status_monitor(message_queue))
    brain_task = asyncio.create_task(robot_brain(message_queue, messages, tools_file))

    while True:
        try:
            await asyncio.sleep(0.1) 
            user_input = await get_async_input("")
            await message_queue.put(user_input)
            
            if user_input.lower() in ['exit', 'quit']:
                break
        except asyncio.CancelledError:
            break

    await brain_task
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

    # Clean up ZMQ context/sockets gracefully
    zmq_req_socket.close()
    zmq_sub_socket.close()
    context.term()

if __name__ == "__main__":
    asyncio.run(main())