import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import plotly.express as px
import sys
import os

def eda_report(data):
    '''Te EDA report will create some files to analyze the in deep the variables of the table.
    The elements will be divided by categoric and numeric and some extra info will printed'''

    describe_result=data.describe()

    eda_path = './files/modeling_output/figures/'
    reports_path='./files/modeling_output/reports/'
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    if not os.path.exists(eda_path):
        os.makedirs(eda_path)

    # Exporting the file
    with open(reports_path+'describe.txt', 'w') as f:
        f.write(describe_result.to_string())
    # Exporting general info
    with open(reports_path+'info.txt','w') as f:
        sys.stdout = f
        data.info()
        sys.stdout = sys.__stdout__

    numeric=['area', 'room', 'price(usd)']

    g=sns.PairGrid(data[numeric])
    g.map(sns.scatterplot)
    g.fig.set_size_inches(12, 12)
    g.fig.suptitle('Relations between features')
    g.savefig(eda_path+'fig1.png')


    # Graficos de correlacion
    corr_df = data[numeric].corr(method="pearson")
    fig1,ax1=plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, ax=ax1, cmap="coolwarm", fmt=".2f")
    fig1.suptitle('Correlation between features')
    fig1.savefig(eda_path+'figcorr2.png')

    
    # Price per neighborhood
    top10_address=data.groupby(['address','room'],as_index=False)['price(usd)'].mean().sort_values(by='price(usd)',ascending=False).head(10)
    top10_address['price(usd)']=top10_address['price(usd)'].round(2)
    fig2=px.bar(top10_address,x='address',y='price(usd)',color='room',text='price(usd)',title='Top 10 prices per address')
    fig2.write_image(eda_path+'fig2.png', format='png', scale=2)
    fig2.write_html(eda_path+'fig2.html') 

    # Area per neighborhood
    top10_address_a=data.groupby(['address','room'],as_index=False)['area'].mean().sort_values(by='area',ascending=False).head(10)
    top10_address_a['area']=top10_address_a['area'].round(2)
    fig2_1=px.bar(top10_address_a,x='address',y='area',color='room',text='area',title='Top 10 areas per address')
    fig2_1.write_image(eda_path+'fig2_1.png', format='png', scale=2)
    fig2_1.write_html(eda_path+'fig2_1.html') 

    # Parking

    ## Price per room

    fig3=px.scatter(data,x='room',y='price(usd)',color='parking',title='Price per room with parking')
    fig3.write_image(eda_path+'fig3.png', format='png', scale=2)
    fig3.write_html(eda_path+'fig3.html')

    ## Price per area

    fig4=px.scatter(data,x='area',y='price(usd)',color='parking',title='Price per area with parking')
    fig4.write_image(eda_path+'fig4.png', format='png', scale=2)
    fig4.write_html(eda_path+'fig4.html') 



    # Warehouse

    ## Price per room

    fig5=px.scatter(data,x='room',y='price(usd)',color='warehouse',title='Price per room with warehouse')
    fig5.write_image(eda_path+'fig5.png', format='png', scale=2)
    fig5.write_html(eda_path+'fig5.html')

    ## Price per area

    fig6=px.scatter(data,x='area',y='price(usd)',color='warehouse',title='Price per area with warehouse')
    fig6.write_image(eda_path+'fig6.png', format='png', scale=2)
    fig6.write_html(eda_path+'fig6.html') 

    # Elevator

    ## Price per room

    fig7=px.scatter(data,x='room',y='price(usd)',color='elevator',title='Price per room with elevator')
    fig7.write_image(eda_path+'fig7.png', format='png', scale=2)
    fig7.write_html(eda_path+'fig7.html')

    ## Price per area

    fig8=px.scatter(data,x='area',y='price(usd)',color='elevator',title='Price per area with elevator')
    fig8.write_image(eda_path+'fig8.png', format='png', scale=2)
    fig8.write_html(eda_path+'fig8.html')  