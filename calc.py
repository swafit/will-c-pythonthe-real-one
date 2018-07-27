import sys

command = sys.argv[1]

if command == "add":
    x = int(sys.argv[2])
    y = int(sys.argv[3])

    print (x+y)

print ("i dont belong here")


if command == "countto":
    x = int(sys.argv[2])

    for i in range(x):
        print(i)


    for item in ['appleas','bananas','oranges']:
        print