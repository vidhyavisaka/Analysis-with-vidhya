# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 10:14:35 2021

@author: ELCOT
"""


#lib

import pandas as pd
import numpy as np
import dash
import dash_html_components as dhc
import dash_core_components as dcc 
import dash_bootstrap_components as dbc
from dash.dependencies import Input,Output,State
import webbrowser
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import os
import wordcloud
from collections import Counter
from wordcloud import WordCloud, STOPWORDS


#gv
project_name= " analysis with vidhya ".title()
app=dash.Dash()
just=["ok","good","bad","very bad",
      'awesome',"i like it a lot"]
#def  

def open_browser():
   webbrowser.open_new('http://127.0.0.1:8050/')  
   
def load_model():
   global br
   #BALANCED REVIEWS OF AMAZON
   br=pd.read_csv("C:/Users/ELCOT/Desktop/amazon 14gb/balanced_reviews.csv")
   br.dropna(inplace=True)
   br=br[br["overall"]!=3]  
   over=br["overall"]
   br["class"]=" "
   br["class"]=np.where(br["overall"]>3,1,0)
   br=br[:100]
   global just_reviews
   just_reviews=br.reviewText[0:100]
   temp = []
   for i in just_reviews:
        temp.append(check_pos_or_neg(i)) 
  
   #SCRAPPED REVIEWS
   global sr     
   sr=pd.read_csv("C:/Users/ELCOT/Desktop/amazon 14gb/scrappedReviews.csv")
   sr=sr[:100]
   sr_temp=[]   
   for j in sr.reviews:
        sr_temp.append(check_pos_or_neg(j)) 
        
   sr['sentiment'] = sr_temp 
   
   pie_val=[]  
   for i in sr["sentiment"]:
      pie_val.append(len(sr[sr.sentiment==1]))
      
      pie_val.append(len(sr[sr.sentiment==0])) 
      break
   
   explode=np.zeros(len(pie_val))
   explode[np.argmax(pie_val)]=0.1
   
   
  
   langs = ['Positive', 'Negative',]
   
   colors = ['#D8BFD8','#E6E6FA']
   plt.pie(pie_val,explode=explode,startangle=90,colors=colors, labels = langs,autopct='%1.2f%%')
  
   cwd = os.getcwd()
   if 'assets' not in os.listdir(cwd):
        os.makedirs(cwd+'/assets')
   plt.savefig('assets/sent.png')
   dataset = sr['reviews'].to_list()
   str1 = ''
   for i in dataset:
        str1 = str1+i
   str1 = str1.lower()

   stopwords = set(STOPWORDS)
   cloud = WordCloud(width = 800, height = 400,background_color ='#E6E6FA',stopwords = stopwords,min_font_size = 10).generate(str1)
   cloud.to_file("assets/wordCloud1.png")
   
def check_pos_or_neg(text): 
   
     global re_mod 
     re_mod=pickle.load(open("model.pkl","rb"))  
     global vocab
     vocab=pickle.load(open("vocab_file.pkl","rb")) 
    
     from sklearn.feature_extraction.text import TfidfVectorizer
     transform=TfidfVectorizer(vocabulary=vocab)
     #global result
     return re_mod.predict(transform.fit_transform([text])) 



def create_ui():
    
    main_layout = dhc.Div( style={"backgroundColor":"#E6E6FA",
                                  "textAlign":"center"},children=[
       dhc.Div(                              
       [
   dhc.Marquee(id="mar",children=[ dhc.H1(id='Main_title', children = "analysis with vidhya".title(),
           style={"color":"#800080",
               "textAlign":"center"})]),
   dhc.Br(),
   dhc.H2(children = "Pie Chart",style = {"color":"#800080",'text-align':'center',"text-decoration":"underline"}),
   dhc.P([dhc.Img(src=app.get_asset_url('sent.png'),style={'width':'700px','height':'400px'})],style={'text-align':'center'}),
   dhc.Hr(style={'background-color':'black'}),
   dhc.Br(),
   dhc.H2(children = "WordCloud",style = {"color":"#800080","text-align":'center','text-decoration':'underline'}),
   dhc.P([dhc.Img(src=app.get_asset_url('wordCloud1.png'),style={'width':'700px','height':'400px'})],style={'text-align':'center'}),
   dhc.Hr(style={'background-color':'black'}),

              
        
        
        
     dcc.Textarea(id="text_input",placeholder="type your review here.....",
                     style={"color":"#4B0082",
                            "textAlign":"left",
                            
                            "backgroundColor":"#DDA0DD",
                            "width":"500px",
                            "height":100
                
                         }
                     
                     
                     
                     ),
        
      dhc.Br(),  
      dcc.Dropdown(id="drop_values",
                     placeholder="select a review",
                     options= [{"label":i,"value":i} for i in sr.reviews],
                     style={"color":"#4B0082",
                            "backgroundColor":"#DDA0DD",
                            "width":"100%",
                            }
        
                               
                     ),
         
        

      dhc.Br(),
      dhc.Button(id="button_review",children="find review".title(),n_clicks=0,
                   style={"color":"#4B0082",
                          "backgroundColor":"#DDA0DD",
                          "width":"500px"
                          
                       } 
                   ),
      dhc.Br(),
        
      dhc.Br(),

       
      dhc.Div(id="res",children= "result"),
      dhc.Div(id="res2",children="result")       
                  
       
        
        
        
        ])])
    return main_layout

@app.callback(
    Output("res","children"), 
     [
     Input("button_review", "n_clicks") 
     ],
     
     [
      State("text_input","value")
      ] 
     )
        

 


def update_app(n_clicks,text_value):
   
    
    if n_clicks >0:
        
      response=check_pos_or_neg(text_value)
      if response[0]==0:
        return dbc.Alert("negative",color="#FA8072")
      if response[0]==1:
        return dbc.Alert("positive",color="#9DC183") 
             
@app.callback(
    Output("res2","children"), 
     [
     Input("button_review", "n_clicks") 
     ],
     
     [
      State("drop_values","value")
      ]
     )
        



def update_app(n_clicks,drop_val):
   
    
    
    if n_clicks >0:
        
      drop_response=check_pos_or_neg(drop_val)
      if drop_response[0]==0:
        return dbc.Alert("negative",color="#FA8072")
      if drop_response[0]==1:
        return dbc.Alert("positive",color="#9DC183")
          
 
 
#main
def main():
    global project_name
    global app
    global br 
    print("start of projet")
    print("my project name:",project_name )
    load_model()
    open_browser()
    
    app.title=project_name
    app.layout=create_ui() 
        
    app.run_server()
    
    
    
    project_name=None
     
    print("end of project")
    
    
if __name__=='__main__' :
    main() 
    
    
    
    
    
    





