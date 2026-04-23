import random
import serial
import threading
from typing import Optional


class UltrasonicSensor:
    """
    Reads real distance data from an ultrasonic module over UART, with
    a simulation/random fallback if the serial port is not available.

    Expected frame format (4 bytes):
        [0] 0xFF (header)
        [1] high byte of distance (mm or mm/10 depending on sensor)
        [2] low byte of distance
        [3] checksum = (byte0 + byte1 + byte2) & 0xFF
    """

    def __init__(
        self,
        port: str = "/dev/ttyTHS1",
        baudrate: int = 9600,
        timeout: float = 0.1,
    ) -> None:
        self._distance_cm: float = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._serial: Optional[serial.Serial] = None

        try:
            self._serial = serial.Serial(port, baudrate, timeout=timeout)
            # Start a background thread that constantly reads sensor data
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
        except Exception as e:
            print(f"[UltrasonicSensor] Failed to open serial port {port}: {e}")
            self._serial = None

    def _reader_loop(self) -> None:
        """Continuously read 4-byte frames from the serial port."""
        if self._serial is None:
            return

        while self._running:
            try:
                data = self._serial.read(4)
                if len(data) != 4:
                    continue

                # data[0] is a byte object; compare with 0xFF
                if data[0] != 0xFF:
                    continue

                distance_raw = (data[1] << 8) + data[2]
                checksum = (data[0] + data[1] + data[2]) & 0xFF

                if checksum != data[3]:
                    continue

                # Many ultrasonic UART modules report distance in mm.
                distance_cm = distance_raw / 10.0

                with self._lock:
                    self._distance_cm = float(distance_cm)

            except Exception as e:
                print(f"[UltrasonicSensor] Read error: {e}")

    @property
    def connected(self) -> bool:
        """Return True if the underlying serial port opened successfully."""
        return self._serial is not None

    def get_distance(self) -> Optional[float]:
        """
        Return the latest distance reading in centimeters.
        If hardware is unavailable, returns a simulated value.
        """
        if self._serial is None:
            # No hardware connected
            return None

        with self._lock:
            if self._distance_cm <= 0:
                # No valid data yet
                return None
            return self._distance_cm

    def close(self) -> None:
        """Cleanly stop the background reader and close the port."""
        self._running = False
        try:
            if self._serial is not None and self._serial.is_open:
                self._serial.close()
        except Exception:
            pass