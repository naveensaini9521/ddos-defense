Run Individual Modes
Normal traffic (60s @ 5 RPS):

bash
python -m simulator.traffic_generator \
 --target http://192.168.100.10/ \
 --mode normal --duration 60 --rps 5
HTTP flood (30s, 50 threads):

bash
python -m simulator.traffic_generator \
 --target http://192.168.100.10/ \
 --mode http_flood --duration 30 --threads 50
HTTP flood with spoofed X-Forwarded-For:

bash
python -m simulator.traffic_generator \
 --target http://192.168.100.10/ \
 --mode spoofed --duration 30 --threads 50
Slowloris (60s, 200 connections):

bash
python -m simulator.traffic_generator \
 --target http://192.168.100.10/ \
 --mode slowloris --duration 60 --connections 200
