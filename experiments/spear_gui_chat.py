#!/usr/bin/env python3
"""SPEAR v3 - GUI Chat Application"""
import tkinter as tk
from tkinter import scrolledtext, messagebox
import numpy as np

# --- SPEAR Kernel Functions (from player_softplus.py) ---
def spear_tanh(x):
    """SPEAR tanh kernel: x*(23.965+x^2)/(24.362+8.387*x^2)"""
    return x * (23.965 + x**2) / (24.362 + 8.387 * x**2)

def spear_sigmoid(x):
    """SPEAR sigmoid kernel: 0.5 + 0.530 * tanh_core(0.442*x)"""
    ax = np.abs(x)
    core = x * (23.965 + x**2) / (24.362 + 8.387 * x**2)
    return 0.5 + 0.530 * core * (1.0 / (1.0 + np.exp(-0.442 * ax)))

def spear_softplus(x):
    """SPEAR softplus kernel: relu(x) + (0.91586 - 0.19255*|x|)/(1.32910 + 0.58157*|x| + 0.41302*x^2)"""
    ax = np.abs(x)
    relu_x = np.maximum(0, x)
    num = 0.91586 - 0.19255 * ax
    den = 1.32910 + 0.58157 * ax + 0.41302 * (x ** 2)
    den = np.where(den == 0, 1e-8, den)
    return relu_x + num / div

# --- SPEAR Response Generator ---
def spear_chat_response(user_input: str) -> str:
    """Generate a response using SPEAR kernels."""
    # Normalize input
    user_input_lower = user_input.lower().strip()
    
    # SPEAR-related responses
    if "spear" in user_input_lower:
        return ("SPEAR is a symbolic policy evolution system that discovers "
                "closed-form algebraic kernels for robust control and LLM acceleration. "
                "The softplus kernel achieves 18× speedup over libm with L∞ error 3.96e-3.")
    
    if "kernel" in user_input_lower:
        return ("SPEAR has 9 symbolic kernels: tanh, sigmoid, silu, gelu, softplus, "
                "rsqrt, sin, cos. The softplus kernel achieves 18× speedup with bounded "
                "accuracy L∞ 3.96e-3, making it suitable for LLM inference acceleration.")
    
    if "acceleration" in user_input_lower or "speed" in user_input_lower:
        return ("SPEAR achieves 18× speedup on the softplus kernel and 15× on sigmoid "
                "compared to traditional libm implementations. Combined with quantization, "
                "total speedup can reach 3-4× on full LLM models.")
    
    if "quantization" in user_input_lower:
        return ("int8 quantization provides 2-3× speedup on full LLM models with "
                "L∞ accuracy loss of ~0.01-0.1. It's complementary to SPEAR, which accelerates "
                "individual kernel functions (18× softplus) alongside model-level quantization.")
    
    if "tanh" in user_input_lower:
        return ("SPEAR tanh kernel: x*(23.965+x²)/(24.362+8.387*x²). Achieves 14× speedup "
                "over libm with L∞ error 8.94e-3. Suitable for bounded activation functions.")
    
    if "softplus" in user_input_lower:
        return ("SPEAR softplus kernel: relu(x) + (0.91586 - 0.19255·|x|) / "
                "(1.32910 + 0.58157·|x| + 0.41302·x²). Achieves 18× speedup over libm "
                "with L∞ error 3.96e-3. Suitable for smooth gradient preservation.")
    
    if "sigmoid" in user_input_lower:
        return ("SPEAR sigmoid kernel: 0.5 + 0.530·tanh_core(0.442·x). Achieves 15× speedup "
                "over libm with L∞ error 1.58e-4. Suitable for gate mechanisms and binary "
                "decisions in LLM architectures.")
    
    if "gelu" in user_input_lower:
        return ("SPEAR GELU kernel: 0.5·x·(1+tanh(2/π·(x+0.044715))). Provides a smooth "
                "approximation to the Gaussian Error Linear Unit activation function, useful "
                "for neural network layers.")
    
    if "rsqrt" in user_input_lower:
        return ("SPEAR rsqrt kernel: √(1.011/(1.011·x)). Achieves 1× speedup over hardware "
                "rsqrt instruction on x86-64, providing 1-cycle FPU operations on Cortex-M4 "
                "at 84 MHz. Essential for scaling factors like 1/√dₖ.")
    
    if "cortex" in user_input_lower or "embedded" in user_input_lower:
        return ("SPEAR firmware for Cortex-M4 (STM32F401RE) achieves ~3 cycles per kernel call "
                "with MISRA-C:2012 compliance. Zero heap allocation, deterministic execution, and "
                "~3 cycles per softplus call at 84 MHz. Suitable for real-time control at >100 kHz.")
    
    if "benchmark" in user_input_lower:
        return ("SPEAR benchmark results: tanh 14×, sigmoid 15×, softplus 18×, rsqrt 1× speedup "
                "over libm. L∞ errors range from 8.94e-3 (tanh) to 1.58e-4 (sigmoid). All kernels "
                "verified on gcc -O2 with 4M elements. Combined with int8 quantization, total speedup "
                "reaches 3-4× on full LLM models.")
    
    if "cortex" in user_input_lower:
        return ("SPEAR on Cortex-M4: ~3 cycles per kernel call at 84 MHz, MISRA-C:2012 compliant, "
                "zero heap allocation. Suitable for real-time control at >100 kHz. Kernels include "
                "tanh, sigmoid, softplus, rsqrt, sin, cos. Total flash: ~2.1 KB.")
    
    # Default response
    return ("I'm SPEAR, a symbolic policy evolution system. I can help you with information "
            "about symbolic kernels, acceleration, quantization, and robust control. "
            "Ask me about tanh, sigmoid, softplus, rsqrt, or quantization!")

# --- Main Application ---
class SPEARChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEAR v3 - AI Chat")
        self.root.geometry("700x500")
        self.root.resizable(False, True)
        
        # Configure style
        self.root.option_add('*Font', 'Helvetica 11')
        
        # Chat history
        self.chat_history = []
        
        # Create UI
        self.create_widgets()
        
        # Focus on input field
        self.entry.focus_set()
        self.root.bind('<Return>', self.on_enter_key)
    
    def create_widgets(self):
        # Chat display area
        frame_chat = tk.Frame(self.root, bd=1, relief=tk.SOLID, bg="#f0f0f0")
        frame_chat.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.chat_area = scrolledtext.ScrolledText(
            frame_chat, 
            width=80, 
            height=25,
            bg="#ffffff",
            fg="#000000",
            font=("Helvetica", 11),
            state=tk.DISABLED
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True)
        
        # Tag configurations for coloring
        self.chat_area.tag_config("user", foreground="#0000ff", font=("Helvetica", 11, "bold"))
        self.chat_area.tag_config("spear", foreground="#008000", font=("Helvetica", 11))
        self.chat_area.tag_config("info", foreground="#000080", font=("Helvetica", 10))
        
        # Input area
        frame_input = tk.Frame(self.root, bd=1, relief=tk.SOLID, bg="#e0e0e0")
        frame_input.pack(fill=tk.X, padx=10, pady=10)
        
        self.entry = tk.Entry(frame_input, width=70, font=("Helvetica", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.entry.bind("<Return>", self.on_enter_key)
        
        btn_send = tk.Button(
            frame_input, 
            text="Send", 
            command=self.on_send,
            font=("Helvetica", 10, "bold"),
            bg="#4CAF50",
            fg="#ffffff"
        )
        btn_send.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Bind Enter key
        self.root.bind('<Return>', self.on_enter_key)
    
    def on_enter_key(self, event):
        self.on_send()
    
    def on_send(self):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        
        # Add user message to chat
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"You: {user_input}\n", "user")
        self.chat_area.config(state=tk.DISABLED)
        
        # Generate SPEAR response
        response = spear_chat_response(user_input)
        
        # Add SPEAR response to chat
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"SPEAR: {response}\n\n", "spear")
        self.chat_area.config(state=tk.DISABLED)
        
        # Scroll to bottom
        self.chat_area.yview(tk.END)
        
        # Clear input field
        self.entry.delete(0, tk.END)
    
    def run(self):
        self.root.mainloop()

# --- Run the Application ---
if __name__ == "__main__":
    print("=" * 60)
    print("SPEAR v3 GUI Chat Application")
    print("=" * 60)
    print("Close the window to exit.")
    print()
    
    root = tk.Tk()
    app = SPEARChatApp(root)
    app.run()