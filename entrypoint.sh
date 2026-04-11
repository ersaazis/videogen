#!/bin/bash

# Start DBUS
mkdir -p /run/dbus
dbus-daemon --system

# Start Warp Service in the background
/usr/bin/warp-svc &

# Wait for warp-svc to be ready
echo "Waiting for warp-svc to start..."
sleep 5

# Set mode to warp (default, but to be sure)
warp-cli --accept-tos mode warp

# Register new account (stateless)
echo "Registering Warp..."
warp-cli --accept-tos registration new

# Connect to Warp
echo "Connecting to Warp..."
warp-cli --accept-tos connect

# Verify connection
echo "Checking Warp status..."
warp-cli --accept-tos status

# Start the application
echo "Starting application..."
exec python app.py
