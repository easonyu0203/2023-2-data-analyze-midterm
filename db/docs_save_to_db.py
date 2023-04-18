import pymongo
import pandas as pd

client = pymongo.MongoClient("mongodb://localhost:27017/")

db = client["my_database"]
collection = db["documents"]
