
#from breezy import timestamp
#import timestamp
!pip install yfinance
import streamlit as st
import yfinance as yf
import pandas as pd
import schedule
import time
import ollama
from datetime import datetime, timedelta


# Fetching historical data for Apple (AAPL) and Dow Jones (DJI)
stock = yf.Ticker("AAPL")
dow_jones = yf.Ticker("^DJI")
data = stock.history(period="1d", interval="1m")
dow_data = dow_jones.history(period="1d", interval="1m")

# Global variables to store rolling data
rolling_window = pd.DataFrame()
dow_rolling_window = pd.DataFrame()
daily_high = float('-inf')
daily_low = float('inf')
buying_momentum = 0
selling_momentum = 0

# Function to process a new stock update every minute
def process_stock_update():
    global rolling_window, data, dow_rolling_window, dow_data
    global daily_high, daily_low, buying_momentum, selling_momentum

    if not data.empty and not dow_data.empty:
        # Simulate receiving a new data point for AAPL and Dow Jones
        update = data.iloc[0].to_frame().T
        dow_update = dow_data.iloc[0].to_frame().T
        data = data.iloc[1:]  # Remove the processed row
        dow_data = dow_data.iloc[1:]

        # Append the new data points to the rolling windows
        rolling_window = pd.concat([rolling_window, update], ignore_index=False)
        dow_rolling_window = pd.concat([dow_rolling_window, dow_update], ignore_index=False)

        # Update daily high and low
        daily_high = max(daily_high, update['Close'].values[0])
        daily_low = min(daily_low, update['Close'].values[0])

        # Calculate momentum
        if len(rolling_window) >= 2:
            price_change = update['Close'].values[0] - rolling_window['Close'].iloc[-2]
            if price_change > 0:
                buying_momentum += price_change
            else:
                selling_momentum += abs(price_change)
                

def calculate_insights(window, dow_window):
    if len(window) >= 5:
        # 5-minute rolling average
        rolling_avg = window['Close'].rolling(window=5).mean().iloc[-1]
        
        # Exponential Moving Average (EMA)
        ema = window['Close'].ewm(span=5, adjust=False).mean().iloc[-1]
        
        # Bollinger Bands (using a 5-period window)
        std = window['Close'].rolling(window=5).std().iloc[-1]
        bollinger_upper = rolling_avg + (2 * std)
        bollinger_lower = rolling_avg - (2 * std)

        # RSI calculation
        delta = window['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean().iloc[-1]
        avg_loss = loss.rolling(window=14, min_periods=1).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else float('nan')
        rsi = 100 - (100 / (1 + rs))

        # Calculate Relative Strength Index (RSI) if there are enough periods (14 is typical)
        delta = window['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean().iloc[-1]
        avg_loss = loss.rolling(window=14, min_periods=1).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else float('nan')
        rsi = 100 - (100 / (1 + rs))

        # Calculate Dow Jones index rolling average
        dow_rolling_avg = dow_window['Close'].rolling(window=5).mean().iloc[-1]
        
        market_open_duration = get_market_open_duration(window)
        if int(market_open_duration) % 5 == 0:  # Trigger LLM every 5 minutes
            get_natural_language_insights(
                rolling_avg, ema, rsi, bollinger_upper, bollinger_lower,
                price_change, volume_change, dow_rolling_avg, market_open_duration, 
                dow_price_change, dow_volume_change, daily_high, daily_low, 
                buying_momentum, selling_momentum
            )
            
def get_natural_language_insights(
    rolling_avg, ema, rsi, bollinger_upper, bollinger_lower,
    price_change, volume_change, dow_rolling_avg, market_open_duration, dow_price_change, dow_volume_change, 
    daily_high, daily_low, buying_momentum, selling_momentum, timestamp
):
    # Vérification et conversion des paramètres en flottants
    try:
        rolling_avg = float(rolling_avg)
        ema = float(ema)
        rsi = float(rsi)
        bollinger_upper = float(bollinger_upper)
        bollinger_lower = float(bollinger_lower)
        price_change = float(price_change)
        dow_rolling_avg = float(dow_rolling_avg)
        market_open_duration = float(market_open_duration)
        dow_price_change = float(dow_price_change)
        daily_high = float(daily_high)
        daily_low = float(daily_low)
        buying_momentum = float(buying_momentum)
        selling_momentum = float(selling_momentum)
    except ValueError as e:
        print("Erreur : un des paramètres n'est pas convertible en nombre.", e)
        return "Erreur : paramètre non valide."

    # Créer le prompt après vérification
    prompt = f"""
    Vous êtes un courtier en bourse professionnel. L'action d'Apple a une moyenne mobile de 5 minutes de {rolling_avg:.2f}.
    La moyenne mobile exponentielle (EMA) est de {ema:.2f}, et l'indice de force relative (RSI) est de {rsi:.2f}.
    Les bandes de Bollinger ont une bande supérieure de {bollinger_upper:.2f} et une bande inférieure de {bollinger_lower:.2f}.
    Le prix a changé de {price_change:.2f}, et le volume a varié de {volume_change}.
    Le prix du Dow Jones a changé de {dow_price_change:.2f}, et le volume a varié de {dow_volume_change}.
    Par ailleurs, l'indice Dow Jones a une moyenne mobile de 5 minutes de {dow_rolling_avg:.2f}.
    Le marché est ouvert depuis {market_open_duration:.2f} minutes.
    Le plus haut d'aujourd'hui était de {daily_high:.2f} et le plus bas de {daily_low:.2f}.
    L'élan d'achat est de {buying_momentum:.2f} et l'élan de vente est de {selling_momentum:.2f}.
    Sur la base de ces données, fournissez des informations sur la tendance actuelle des actions et le sentiment général du marché.
    Les informations ne doivent pas dépasser 100 mots et ne doivent pas avoir d'introduction.
    """
    
    # Appel de l'API avec la génération de la réponse
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extraction et affichage de la réponse
    response_text = response['message']['content'].strip()
    print("Insight en langage naturel :", response_text)
    return response_text

response_text = get_natural_language_insights(
    rolling_avg=150.34,
    ema=152.45,
    rsi=55.6,
    bollinger_upper=155.3,
    bollinger_lower=145.2,
    price_change=1.23,
    volume_change=1000,
    dow_rolling_avg=34000.12,
    market_open_duration=120,
    dow_price_change=300.4,
    dow_volume_change=2000,
    daily_high=153.45,
    daily_low=149.5,
    buying_momentum=60.5,
    selling_momentum=40.3,
    timestamp="2024-11-08 15:30:00"
)


# Utilisation de response_text dans Streamlit
if response_text:
    message.write(response_text)
else:
    message.write("Erreur lors de la génération de l'insight.")


message = st.chat_message("assistant")
message.write(timestamp)
message.write(response_text)