import pickle
import objectToSerialize
"""
arnost = objectToSerialize.objectToSerialize("Arnost", "518329323")

f = open("arnost.bn", "wb")
pickle.dump(arnost, f)
"""
f = open("arnost.bn" ,"rb")
arnost = pickle.load(f)
arnost.toString()
