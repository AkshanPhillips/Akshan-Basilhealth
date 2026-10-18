# Troubleshooting: SP2 Smart Plug Shows Offline

1. The SP2 only supports 2.4GHz Wi-Fi networks. If your router broadcasts a combined SSID for 2.4GHz
   and 5GHz, the plug may have associated with the 5GHz band and will never connect — create a
   separate 2.4GHz-only SSID temporarily for setup if needed.
2. Power-cycle the plug (unplug for 10 seconds) and check that the LED is fast-blinking blue (setup mode)
   before attempting to reconnect it in the app.
3. Reset the plug by holding the button for 5 seconds until the LED blinks rapidly, then redo setup from
   scratch in the NovaVolt Home app.
4. Router-side MAC filtering or guest-network client isolation can block the plug even after it joins
   Wi-Fi successfully — check router settings if the plug connects but still shows "offline" in-app.
5. If the plug repeatedly drops offline after a successful setup (not a setup failure), this is likely a
   hardware fault and should be escalated for a warranty/replacement check rather than repeated resets.
