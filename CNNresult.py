from setup import *
from CNNmodel import create_model
import globalVariable


date_format = "%Y-%m-%d"
date_now = dt.datetime.now().strftime('%Y-%m-%d')
date_2_month_back = (dt.date.today() - dt.timedelta(days=100)).strftime('%Y-%m-%d')
end_date = date_now
end_date_time = datetime.strptime(end_date, date_format).date()
date_1_month_back = (end_date_time - dt.timedelta(days=120)).strftime('%Y-%m-%d')


model = create_model()
model.load_weights('model/cnn.h5')

file_path = "numbers.txt" 
results = {}

def images_convertor(paths):
    image = Image.open(paths)
    image = image.resize((150, 150))
    image = image.convert('RGB')  
    image = np.array(image) / 255.0
    return image

with open(file_path, "r") as file:
    
    for line in file:
        ticker_symbol=line.strip()
        save_path = 'stock/real_time_image/'+ticker_symbol.split(".")[0]
       
        data = yf.download(ticker_symbol, start=date_1_month_back, end=end_date)
        mpf.plot(data, type='candle', 
                 ylabel='Price', style='yahoo', axisoff=True, savefig=save_path)
        image_path = save_path+".png"
        image = images_convertor(image_path)  
        image = np.expand_dims(image, axis=0)  
        predictions = model.predict(image)

        class_probabilities = np.round(predictions[0],3)
        class_labels = globalVariable.categories
        default_category = 'unknown' 
        if not np.any(class_probabilities > 0.5):
            predicted_class_label = default_category
        else: 
            predicted_class_index = np.argmax(class_probabilities)
            predicted_class_label = class_labels[predicted_class_index]
            results[ticker_symbol] = predicted_class_label  
            print(ticker_symbol)

with open("CNNresults.json", "w") as file:
    json.dump(results, file)       
