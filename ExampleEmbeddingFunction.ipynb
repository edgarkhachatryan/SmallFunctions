#Creating a model that maps categorical variables to the combination of numeric colums. train test merged.

input_train_test=[]
output_train_test_embeddings=[]

#numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

for categorical_var in embed_cols:
    
    #Name of the categorical variable that will be used in the Keras Embedding layer
    cat_emb_name= categorical_var.replace(" ", "")+'_Embedding' #If contains space delete and concate '_Embedding'
  
    # Define the embedding_size at least 50% of unique values of combined df of train and test.
    no_of_unique_cat  = pd.concat([weather_train, weather_test])[categorical_var].nunique()
    embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 50 ))
    
    #One Embedding Layer for each categorical variable
    input_model = Input(shape=(1,))
    output_model = Embedding(no_of_unique_cat, embedding_size, name=cat_emb_name)(input_model)
    output_model = Reshape(target_shape=(embedding_size,))(output_model)
    
    #Appending all the categorical inputs
    input_train_test.append(input_model) 
    
    #Appending all the embeddings
    output_train_test_embeddings.append(output_model) 
    
output = Concatenate()(output_train_test_embeddings)
