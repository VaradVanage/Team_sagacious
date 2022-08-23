from matplotlib.pyplot import pause
from googlesearch import search
query = input("Enter query:")
for i in search(query, tld = "com", num=10, stop=10, pause=2):
    print(i)