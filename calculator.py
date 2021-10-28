col = int(input('no of cols: '))
filter = int(input('size of filter: '))
stride = int(input('size of stride: '))

idx = filter
i = 0
while True:
    print(idx)
    print()
    i = i + 1
    print(i)
    print()
    idx = idx + stride
    if idx >= col :
        print(f"Your input col dimension is {col}...\nand the kernel size is {filter}... \nand the stride is {stride}...")
        print(f"end at {i}... This is the next layer input col dimension...")
        break 