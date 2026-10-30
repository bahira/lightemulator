#!/usr/bin/env python3
"""Test SPEAR API Response Generation"""
import sys
sys.path.insert(0, '/Users/Yuri/Documents/lightemulator')

from spear_api_server import generate_spear_response

# Test various inputs
test_inputs = [
    'hello',
    'how are you', 
    'what is your name',
    'acceleration',
    'kernel',
    'quantization',
    'speed',
    'default topic',
]

print('SPEAR Response Test:')
print('=' * 50)
for user_input in test_inputs:
    response = generate_spear_response(user_input)
    print(f'User: "{user_input}"')
    print(f'SPEAR: "{response}"')
    print()