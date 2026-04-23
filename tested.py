import serial

ser = serial.Serial('/dev/ttyTHS1',9600,timeout=1)

while True:
    data = ser.read(4)

    if len(data) == 4 and data[0] == 0xFF:
        distance = (data[1] << 8) + data[2]
        checksum = (data[0] + data[1] + data[2]) & 0xFF

        if checksum == data[3]:
            print("Distance:", distance/10, "cm")