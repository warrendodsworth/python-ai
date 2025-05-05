import json

# File reading and writing
data = {'name': 'Warren Dodsworth'}

# Write
# regular way
# f = open('data.json', 'w')
# t = json.dump(data, indent=2, sort_keys=True)
# f.write(t)
# f.close()

# using with, so it auto closes the file
with open('data.json', 'w') as file:
    json.dump(data, file, indent=2)


# Read 
# using with
with open('data.json', 'r') as file: 
    data = json.load(file)

print(data)

# regular way 
# f = open('data.json', 'r')
# print(f.readlines())
# f.close()

# Http Requests
# header, output = client.request('', method="GET", body=None,  headers=None)
