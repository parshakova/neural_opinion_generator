import ijson,os

mdata = []
count =0

newpath = r'json_data_250' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
    

    
index = 0
with open('cmv_20161111.jsonlist') as file:
    for line in file:
        count +=1
        mdata.append(json.loads(line.decode('utf-8')))
        if count > 250:
            print(index)
            count = 0
            with open(os.path.join(newpath,str(index)+'.jsonlist'), mode='w') as outfile:
                json.dump(mdata, outfile)
            index += 1
            del mdata
            mdata = []