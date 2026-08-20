import asyncio
import json
import zmq
import zmq.asyncio
from openai import AsyncOpenAI
from utilities.database_functions import load_json, update_memory, build_system_prompt

client = AsyncOpenAI()
context = zmq.asyncio.Context()

zmq_req_socket = context.socket(zmq.REQ)
zmq_req_socket.connect("tcp://localhost:5555")

zmq_sub_socket = context.socket(zmq.SUB)
zmq_sub_socket.connect("tcp://localhost:5556")
zmq_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
zmq_req_lock = asyncio.Lock()

AI_MODEL = "gpt-5.4-mini"
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
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)

async def handle_tool_calls(item):
    if item.name == "update_memory":
        args = json.loads(item.arguments)
        print(f"[System: Saving to disk -> {args.get('category')}: {args.get('new_info')}]")
        result = update_memory(MEMORY_PATH, args.get("category"), args.get("new_info"))
        return {"type": "function_call_output", "call_id": item.call_id, "output": result}

    elif item.name in ["blind_move", "navigate_to_pose", "stop_motors", "get_vision","find_object"]:
        args = json.loads(item.arguments) if item.arguments else {}
        payload = {"command": item.name, **args}
        
        print(f"[System: Dispatching {item.name} command to ROS 2...]")
        print(f"Payload: {payload}")
        async with zmq_req_lock:
            await zmq_req_socket.send_json(payload)
            feedback = await zmq_req_socket.recv_json()
        print(f"[System: ROS 2 Acknowledged: {feedback['status']}]")
        
        return {"type": "function_call_output", "call_id": item.call_id, "output": json.dumps(feedback)}
    
    return None

async def robot_brain(message_queue, messages, tools_file):
    while True:
        user_input = await message_queue.get()
        
        if user_input.lower() in ['exit', 'quit']:
            print("🤖 DJ: Powering down. See you later!")
            break

        messages.append({"role": "user", "content": user_input})

        while True:
            response = await client.responses.create(
                model=AI_MODEL,
                tools=tools_file,
                input=messages,
            )

            messages += response.output
            tool_called = False
            
            tool_tasks = []
            for item in response.output:
                if item.type == "function_call":
                    tool_called = True
                    tool_tasks.append(handle_tool_calls(item))
            
            if tool_called:
                tool_results = await asyncio.gather(*tool_tasks)
                messages.extend([res for res in tool_results if res is not None])
            else:
                break
                
        print(f"\n🤖 DJ: {response.output_text}\n")

async def main():
    identity_file = load_json(IDENTITY_PATH)
    memory_file = load_json(MEMORY_PATH)
    
    tools_file = []
    for path in TOOLS_PATHS:
        tools_file.extend(load_json(path))

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