import requests
from bs4 import BeautifulSoup
import re
import os
import time
from flask import Flask, render_template, request

app = Flask(__name__)
now = 0
prev_item = ""

@app.route('/', methods=['POST','GET'])
def index():
  global prev_item
  global now
  global prev_count
  result = {"answer":0, "time":0}
  if request.method=='POST':
    search_item = {"search": request.form["search"]}
    if search_item["search"].lower() != prev_item:
      now = time.time()
    prev_item = search_item["search"].lower()
    result = search(search_item)
    result["time"] = "{:.2f} seconds".format(time.time() - now)
  return render_template('index.html', count=result)

def search(item):
  result = 0
  url = "https://en.wikipedia.org/wiki/Black_Panther_(film)"

  for i in range(10):
    request = requests.get(url)
    soup = BeautifulSoup(request.text, 'html.parser')
    url = "https://en.wikipedia.org/%s" % soup.find("a", href=True)['href']
    text = soup.text.lower()

    # YOUR CODE HERE: Find occurences of word here.
    result += len(re.findall(item['search'].lower(), text))

  return {
    "answer":
    result
  }

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8080)
