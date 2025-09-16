from __future__ import annotations
class IBClient:
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
    def connect(self):
        self.connected = True
    def disconnect(self):
        self.connected = False
    def place_pair_order(self, symbol_y: str, qty_y: int, side_y: int, symbol_x: str, qty_x: int, side_x: int):
        if not self.connected:
            raise RuntimeError('IBClient non connecté.')
        return {'y': {'symbol': symbol_y, 'qty': qty_y, 'side': side_y}, 'x': {'symbol': symbol_x, 'qty': qty_x, 'side': side_x}, 'status': 'SIMULATED_SENT'}
