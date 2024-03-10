from typing import Literal
from sys import argv
from src.position import Position
from pprint import pprint

# id_bus, line, x, y, t(sec)
type db_entry = tuple[str, int, Position, int]

def total(snapshots: list[tuple[Position, int]], measure: Literal['time', 'distance']) -> float:
    sorted_snaps = sorted(snapshots, key=lambda s: s[1])
    if measure == 'time':
        return sorted_snaps[-1][1] - sorted_snaps[0][1]
    time_sequence: list[Position] = [snap[0] for snap in sorted_snaps]
    previous_position: Position | None = None
    total = 0
    for position in time_sequence:
        if not previous_position:
            previous_position = position
            continue
        total += previous_position.distance(position)
        previous_position = position
    return total

def bus_total_distance(db: list[db_entry], id_bus: str) -> float:
    return total([(entry[2], entry[3]) for entry in db if entry[0] == id_bus], 'distance')

def line_average_speed(db: list[db_entry], line_id: int) -> float:
    bus_snaps = {}
    for entry in db:
        id_bus, line, position, time = entry
        if line != line_id:
            continue
        snaps: list = bus_snaps.get(id_bus, [])
        snaps.append((position, time))
        bus_snaps[id_bus] = snaps
    bus_distances = [total(snaps, 'distance') for snaps in bus_snaps.values()]
    bus_time_intervals = [total(snaps, measure='time') for snaps in bus_snaps.values()]
    return sum(bus_distances) / sum(bus_time_intervals)

def main():
    db: list[db_entry] = []

    file_name, flag, input = argv[1:]
    with open(file_name, 'r') as file:
        for file_line in file:
            tokens = file_line.split(' ')
            id_bus, bus_line, x, y, t = tokens
            bus_line = int(bus_line)
            x = float(x)
            y = float(y)
            position = Position(x, y)
            t = int(t)
            entry = id_bus, bus_line, position, t
            db.append(entry)

    if flag == '-b':
        total_distance = bus_total_distance(db, input)
        print(f'{input}, - Total Distance: {total_distance}')
    elif flag == '-l':
        average_speed = line_average_speed(db, int(input))
        print(f'{input} - Avg Speed: {average_speed}')

if  __name__ == '__main__':
    main()