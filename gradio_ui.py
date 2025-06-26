import requests
import gradio as gr
import json

BACKEND_URL = "http://localhost:8000/chat"

def test_backend_connection():
    """Test if backend is reachable"""
    try:
        health_url = "http://localhost:8000/health"
        response = requests.get(health_url) 
        print(f"Backend health check: {response.status_code} - {response.json()}")
        return True
    except Exception as e:
        print(f"Backend connection failed: {e}")
        return False

def chat_fn(message, history):
    """Chat function for Gradio interface"""
    try:
        print(f"Sending message: {message}")
        
        response = requests.post(
            BACKEND_URL, 
            json={"message": message}, 
            headers={"Content-Type": "application/json"}
        )  
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Response JSON: {result}")
                answer = result.get("answer", "⚠️ No answer in response")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw response: {response.text}")
                answer = f"⚠️ Invalid JSON response: {response.text[:200]}"
        else:
            print(f"HTTP Error: {response.status_code}")
            print(f"Response text: {response.text}")
            answer = f"⚠️ Backend error (HTTP {response.status_code}): {response.text[:200]}"
        
        print(f"Final answer: {answer}")
        return answer
        
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to backend. Make sure the backend server is running on localhost:8000"
    except Exception as e:
        error_msg = f"⚠️ Frontend error: {str(e)}"
        print(error_msg)
        return error_msg

print("Testing backend connection...")
if test_backend_connection():
    print("Backend is reachable")
else:
    print("Backend is not reachable")

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Lab Help Bot",
    description="qwen2.5vl:3b",
    type="messages"
)

if __name__ == "__main__":
    print("Starting frontend server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )