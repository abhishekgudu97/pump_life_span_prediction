import joblib
import pandas as pd

# Load models function
def load_models():
    model_paths = [r"C:\Users\Abhishek\Desktop\project done\saptarshi\updated\top_1_model_Gradient Boosting Regressor.pkl", 
                   r"C:\Users\Abhishek\Desktop\project done\saptarshi\updated\top_2_model_Random Forest Regressor.pkl", 
                   r"C:\Users\Abhishek\Desktop\project done\saptarshi\updated\top_3_model_K-Nearest Neighbors Regressor.pkl"]
    models = [joblib.load(path) for path in model_paths]
    return models

# Preprocess the input for the model
def preprocess_input(pump_power_hp, working_hours_daily, power_fluctuation, mechanical_loss, environmental_loss):
    input_df = pd.DataFrame({
        'Motor_Power_HP': [pump_power_hp],
        'Working_Hours_Per_Day': [working_hours_daily],
        'Power_Fluctuation': [power_fluctuation],
        'Mechanical_Loss': [mechanical_loss],
        'Environmental_Loss': [environmental_loss],
        'Electrical_Loss': [5]  # Electrical loss is fixed at 5%
    })
    return input_df

# Predict the lifespan using the models
def predict_pump_lifespan(models, pump_power_hp, working_hours_daily, power_fluctuation, mechanical_loss, environmental_loss):
    input_df = preprocess_input(pump_power_hp, working_hours_daily, power_fluctuation, mechanical_loss, environmental_loss)
    predictions = []
    
    for model in models:
        predicted_lifespan = model.predict(input_df)
        predictions.append(predicted_lifespan[0])
    
    return predictions

# Main function to run the prediction
def run_pump_prediction():
    models = load_models()
    
    # Get user input
    pump_power_hp = float(input("Enter pump power in HP: "))
    working_hours_daily = float(input("Enter working hours per day: "))
    power_fluctuation = float(input("Enter power fluctuation (0-20%): "))
    mechanical_loss = float(input("Enter mechanical loss (0-2%): "))
    environmental_loss = float(input("Enter environmental loss (0-0.5%): "))
    
    # Predict using the top 3 models
    predictions = predict_pump_lifespan(models, pump_power_hp, working_hours_daily, power_fluctuation, mechanical_loss, environmental_loss)
    
    # Display predictions
    print("\nPredicted Lifespans based on your input:")
    for i, pred in enumerate(predictions, 1):
        print(f"Model {i} Prediction: {pred:.2f} hours")
    
    # Average prediction
    avg_lifespan = sum(predictions) / len(predictions)
    print(f"\nThe average predicted pump lifespan is: {avg_lifespan:.2f} hours")
    print(f"Equivalent to approximately {avg_lifespan / (working_hours_daily * 365):.2f} years.")

# Entry point
if __name__ == "__main__":
    run_pump_prediction()
