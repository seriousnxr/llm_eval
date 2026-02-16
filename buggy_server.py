#!/usr/bin/env python3

import json
import random
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta
import threading
import hashlib


class RateLimiter:
    def __init__(self, limit=50, window_seconds=60):
        self.limit = limit
        self.window_seconds = window_seconds
        self.remaining = limit
        self.reset_time = datetime.now() + timedelta(seconds=window_seconds)
        self.lock = threading.Lock()
        self.cooldown_until = None
    
    def check(self):
        with self.lock:
            now = datetime.now()
            
            if now >= self.reset_time:
                self.remaining = self.limit
                self.reset_time = now + timedelta(seconds=self.window_seconds)
                self.cooldown_until = None
            
            if self.cooldown_until and now < self.cooldown_until:
                return False, 0, int(self.reset_time.timestamp())
            
            if self.remaining > 0:
                self.remaining -= 1
                if self.remaining == 0:
                    self.cooldown_until = now + timedelta(seconds=5)
                return True, self.remaining, int(self.reset_time.timestamp())
            else:
                return False, 0, int(self.reset_time.timestamp())


rate_limiter = RateLimiter()


def generate_response(prompt, model="gpt-4"):
    is_math = "boxed" in prompt.lower() or "math" in prompt.lower()
    is_code = "def " in prompt or "write a function" in prompt.lower() or "implement" in prompt.lower()
    is_mcq = any(f"({letter})" in prompt or f"{letter}." in prompt or f"{letter})" in prompt 
                 for letter in "ABCD")
    
    if is_mcq:
        answer_letter = random.choice(["A", "B", "C", "D"])
        responses = [
            f"The answer is {answer_letter}.",
            f"I believe the correct answer is ({answer_letter}).",
            f"{answer_letter}",
            f"The correct option is {answer_letter}.",
            f"After careful consideration, I choose {answer_letter}.",
        ]
        content = random.choice(responses)
    
    elif is_math:
        answer_value = random.choice([
            str(random.randint(1, 100)),
            f"{random.randint(1, 50)}/{random.randint(2, 20)}",
            f"{random.random() * 100:.2f}",
            f"\\frac{{{random.randint(1, 10)}}}{{{random.randint(2, 10)}}}",
        ])
        
        use_standard_format = random.random() > 0.20
        if use_standard_format:
            content = f"Let me solve this step by step.\n\nTherefore, the answer is \\boxed{{{answer_value}}}."
        else:
            alt_formats = [
                f"The answer is {answer_value}.",
                f"Therefore, the solution is {answer_value}.",
                f"So we get {answer_value}.",
                f"Final answer: {answer_value}",
                answer_value,
            ]
            content = random.choice(alt_formats)
    
    elif is_code:
        content = """def solution(nums):
    result = []
    for num in nums:
        result.append(num * 2)
    return result"""
    
    else:
        content = "I understand your question. Here is my response based on the context provided."
    
    return {
        "id": f"chatcmpl-{random.randint(100000, 999999)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()) * 2,
            "completion_tokens": len(content.split()) * 2,
            "total_tokens": (len(prompt.split()) + len(content.split())) * 2
        }
    }


class ServerHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {format % args}")
    
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404, "Not Found")
            return
        
        load_factor = random.random()
        if load_factor < 0.08:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": {
                    "message": "The server is currently experiencing high load. Please try again.",
                    "type": "server_error",
                    "code": 503
                }
            }).encode())
            return
        
        allowed, remaining, reset_time = rate_limiter.check()
        
        if not allowed:
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-RateLimit-Limit", "50")
            self.send_header("X-RateLimit-Remaining", "0")
            self.send_header("X-RateLimit-Reset", str(reset_time))
            self.send_header("Retry-After", "5")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": {
                    "message": "Rate limit exceeded. Please retry after a few seconds.",
                    "type": "rate_limit_error",
                    "code": 429
                }
            }).encode())
            return
        
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        
        try:
            request_data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        
        messages = request_data.get("messages", [])
        model = request_data.get("model", "gpt-4")
        
        if not messages:
            self.send_error(400, "Missing messages")
            return
        
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break
        
        response_data = generate_response(prompt, model=model)
        response_json = json.dumps(response_data)
        
        request_hash = hashlib.md5(prompt.encode()).hexdigest()
        hash_val = int(request_hash[:8], 16)
        should_truncate = (hash_val % 50) == 0
        
        if should_truncate:
            cutoff = len(response_json) // 2 + random.randint(0, len(response_json) // 4)
            response_json = response_json[:cutoff]
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-RateLimit-Limit", "50")
        self.send_header("X-RateLimit-Remaining", str(remaining))
        self.send_header("X-RateLimit-Reset", str(reset_time))
        self.end_headers()
        
        processing_time = random.random()
        if processing_time < 0.05:
            chunk_size = max(10, len(response_json) // 5)
            for i in range(0, len(response_json), chunk_size):
                chunk = response_json[i:i + chunk_size]
                self.wfile.write(chunk.encode())
                self.wfile.flush()
                if i + chunk_size < len(response_json):
                    time.sleep(2.0 + random.random())
        else:
            self.wfile.write(response_json.encode())
    
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
        else:
            self.send_error(404, "Not Found")


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    args = parser.parse_args()
    
    server = HTTPServer((args.host, args.port), ServerHandler)
    print(f"=" * 70)
    print(f"OpenAI-Compatible API Server")
    print(f"=" * 70)
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"Endpoint: POST /v1/chat/completions")
    print(f"Health check: GET /health")
    print(f"")
    print(f"Rate limiting: 50 requests per minute")
    print(f"=" * 70)
    print(f"")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
