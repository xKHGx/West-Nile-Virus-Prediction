# Data Analysis
import pandas as pd
import numpy as np
from scipy import stats

# Data Visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set()

# Geographical Plotting
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

# Data Grouping
from sklearn.cluster import KMeans

# Regression Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Classification Models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# Hyper-parameter Tuning
from sklearn.model_selection import GridSearchCV

# Sklearn Data Pre-processing
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Resample / Re-weight for Unbalanced Data Set
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Model Evaluation Tools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

# Feature Contribution
import shap

# Save Model Coefficients
import joblib

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Misc
import os
import textwrap
import re




# Helper function to fill missing values with monthly median
def fill_with_monthly_median(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df[column] = df.groupby('Month')[column].transform(lambda x: x.fillna(x.median()))
    return df

# Helper function to recode CodeSum values to unique numerical identifiers
def recode_codesum(codesum,codesum_to_id):
    codes_in_row = codesum.split()
    unique_ids = set()
    for code in codes_in_row:
        if code in codesum_to_id:
            unique_ids.add(codesum_to_id[code])
        elif code in ['BCFG', 'MIFG', 'VCFG']:
            unique_ids.add(codesum_to_id['FG'])
        elif code in ['TSRA', 'VCTS']:
            unique_ids.add(codesum_to_id['TS'])
    return sum(unique_ids)


# Helper function to wrap text for x-axis labels and table headers
def wrap_text(text, width):
    return "\n".join(textwrap.wrap(text, width))

# Helper function to generate the same type of chart for different features
def plot_features_by_group(df, group_col, feature_1, feature_2, y2_limit=None, agg_func='sum', color_1='blue', color_2='green'):
    # Group the data by the specified column and aggregate feature_1 and feature_2
    if agg_func == 'sum':
        grouped_df = df.groupby([group_col]).agg({feature_1: 'sum', feature_2: 'sum'}).reset_index()
    elif agg_func == 'mean':
        grouped_df = df.groupby([group_col]).agg({feature_1: 'mean', feature_2: 'mean'}).reset_index()
    elif agg_func == 'value':
        grouped_df = df[[group_col, feature_1, feature_2]].copy()
    else:
        raise ValueError("agg_func should be one of 'sum', 'mean', or 'value'")
    
    # Create figure and axis with increased height
    fig, ax1 = plt.subplots(figsize=(14, 12))

    # Plot the feature_1 by the grouping column as a bar chart on ax1
    groups = grouped_df[group_col]
    feature_1_values = grouped_df[feature_1]
    feature_2_values = grouped_df[feature_2]

    # Wrap text for x-axis labels
    wrapped_labels = [wrap_text(str(label), 5) for label in groups]

    ax1.bar(wrapped_labels, feature_1_values, color=color_1, label=f'{agg_func.capitalize()} {feature_1}', alpha=0.6)

    ax1.set_xlabel(group_col.capitalize(), fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'{agg_func.capitalize()} {feature_1}', fontsize=14, fontweight='bold', color=color_1)
    ax1.tick_params(axis='y', labelcolor=color_1)
    ax1.grid(True)

    # Create a twin Axes sharing the x-axis for the line chart
    ax2 = ax1.twinx()

    # Plot feature_2 as a secondary y-axis line chart
    ax2.plot(wrapped_labels, feature_2_values, color=color_2, label=f'{agg_func.capitalize()} {feature_2}', linestyle='-', marker='o')

    ax2.set_ylabel(f'{agg_func.capitalize()} {feature_2}', fontsize=14, fontweight='bold', color=color_2)
    ax2.tick_params(axis='y', labelcolor=color_2)

    # Optionally set the y-axis limit for feature_2
    if y2_limit:
        ax2.set_ylim(y2_limit[0], y2_limit[1])

    # Add a title to the plot
    plt.title(f'{agg_func.capitalize()} {feature_1} and {feature_2} by {group_col.capitalize()}', fontsize=16, fontweight='bold')

    # Add legends to both axes
    ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 0.4), fontsize=12)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=12)

    # Prepare data for the feature_1 table
    table_data_feature_1 = grouped_df[[group_col, feature_1]].set_index(group_col).T

    # Wrap text for table headers
    table_data_feature_1.columns = [wrap_text(str(col), 5) for col in table_data_feature_1.columns]

    # Add a label for the feature_1 table
    plt.text(0.5, -0.15, f'{agg_func.capitalize()} {feature_1} by {group_col.capitalize()}', fontsize=14, fontweight='bold', ha='center', transform=ax1.transAxes)

    # Add a data point table for feature_1 below the plot
    if not table_data_feature_1.empty:
        table1 = plt.table(cellText=table_data_feature_1.values,
                          rowLabels=table_data_feature_1.index,
                          colLabels=table_data_feature_1.columns,
                          loc='bottom',
                          bbox=[0, -0.5, 1, 0.2],  # Adjusted bbox for larger table
                          cellLoc='center')

        # Format the table
        table1.auto_set_font_size(False)
        table1.set_fontsize(12)
        table1.scale(1.2, 1.5)  # Adjusted scale for bigger row height

        # Bold the fonts in the table
        for key, cell in table1.get_celld().items():
            cell.set_text_props(fontweight='bold')

    # Prepare data for the feature_2 table
    table_data_feature_2 = grouped_df[[group_col, feature_2]].set_index(group_col).T

    # Wrap text for table headers
    table_data_feature_2.columns = [wrap_text(str(col), 5) for col in table_data_feature_2.columns]

    # Add a label for the feature_2 table
    plt.text(0.5, -0.55, f'{agg_func.capitalize()} {feature_2} by {group_col.capitalize()}', fontsize=14, fontweight='bold', ha='center', transform=ax1.transAxes)

    # Add a data point table for feature_2 below the plot
    if not table_data_feature_2.empty:
        table2 = plt.table(cellText=table_data_feature_2.values,
                           rowLabels=table_data_feature_2.index,
                           colLabels=table_data_feature_2.columns,
                           loc='bottom',
                           bbox=[0, -0.75, 1, 0.2],  # Adjusted bbox for larger table
                           cellLoc='center')

        # Format the table
        table2.auto_set_font_size(False)
        table2.set_fontsize(12)
        table2.scale(1.2, 1.5)  # Adjusted scale for bigger row height

        # Bold the fonts in the table
        for key, cell in table2.get_celld().items():
            cell.set_text_props(fontweight='bold')

    plt.subplots_adjust(left=0.1, bottom=0.6, right=0.85, top=0.9)

    # Adjust layout for better fit
    plt.tight_layout()

    # Show the plot
    plt.show()


# Helper function to wrap text for x-axis labels and table headers
def wrap_text(text, width):
    return "\n".join(textwrap.wrap(text, width))

# Helper function to generate the same type of chart for different features
def plot_features_by_month_and_year(df, month_col, year_col, feature_1, feature_2, y2_limit=None, agg_func='sum', color_1='blue', color_2='green'):
    # Group the data by month and year and aggregate feature_1 and feature_2
    if agg_func == 'sum':
        grouped_df = df.groupby([year_col, month_col]).agg({feature_1: 'sum', feature_2: 'sum'}).reset_index()
    elif agg_func == 'mean':
        grouped_df = df.groupby([year_col, month_col]).agg({feature_1: 'mean', feature_2: 'mean'}).reset_index()
    elif agg_func == 'value':
        grouped_df = df[[year_col, month_col, feature_1, feature_2]].copy()
    else:
        raise ValueError("agg_func should be one of 'sum', 'mean', or 'value'")

    # Pivot the data to prepare for plotting
    pivot_feature_1 = grouped_df.pivot(index=month_col, columns=year_col, values=feature_1).fillna(0)
    pivot_feature_2 = grouped_df.pivot(index=month_col, columns=year_col, values=feature_2).fillna(0)

    # Create figure and axis with increased height
    fig, ax1 = plt.subplots(figsize=(14, 12))

    # Plot the stacked bar chart for feature_1 by month and year on ax1
    months = grouped_df[month_col].unique()
    years = grouped_df[year_col].unique()

    bottom = np.zeros(len(months))
    for year in years:
        ax1.bar(months, pivot_feature_1[year], bottom=bottom, label=f'{year}', alpha=0.6)
        bottom += pivot_feature_1[year]

    ax1.set_xlabel(month_col.capitalize(), fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'{agg_func.capitalize()} {feature_1}', fontsize=14, fontweight='bold', color=color_1)
    ax1.tick_params(axis='y', labelcolor=color_1)
    ax1.grid(True)

    # Create a twin Axes sharing the x-axis for the line chart
    ax2 = ax1.twinx()

    # Plot feature_2 as a line chart for different years
    for year in years:
        ax2.plot(months, pivot_feature_2[year], label=f'{year} {feature_2}', linestyle='-', marker='o')

    ax2.set_ylabel(f'{agg_func.capitalize()} {feature_2}', fontsize=14, fontweight='bold', color=color_2)
    ax2.tick_params(axis='y', labelcolor=color_2)

    # Optionally set the y-axis limit for feature_2
    if y2_limit:
        ax2.set_ylim(y2_limit[0], y2_limit[1])

    # Add a title to the plot
    plt.title(f'{agg_func.capitalize()} {feature_1} and {feature_2} by {month_col.capitalize()} and {year_col.capitalize()}', fontsize=16, fontweight='bold')

    # Add legends to both axes
    ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 0.4), fontsize=12)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=12)

    # Prepare data for the feature_1 table
    table_data_feature_1 = pivot_feature_1.T

    # Wrap text for table headers
    table_data_feature_1.columns = [wrap_text(str(col), 5) for col in table_data_feature_1.columns]

    # Add a label for the feature_1 table
    plt.text(0.5, -0.15, f'{agg_func.capitalize()} {feature_1} by {month_col.capitalize()} and {year_col.capitalize()}', fontsize=14, fontweight='bold', ha='center', transform=ax1.transAxes)

    # Add a data point table for feature_1 below the plot
    if not table_data_feature_1.empty:
        table1 = plt.table(cellText=table_data_feature_1.values,
                          rowLabels=table_data_feature_1.index,
                          colLabels=table_data_feature_1.columns,
                          loc='bottom',
                          bbox=[0, -0.5, 1, 0.2],  # Adjusted bbox for larger table
                          cellLoc='center')

        # Format the table
        table1.auto_set_font_size(False)
        table1.set_fontsize(12)
        table1.scale(1.2, 1.5)  # Adjusted scale for bigger row height

        # Bold the fonts in the table
        for key, cell in table1.get_celld().items():
            cell.set_text_props(fontweight='bold')

    # Prepare data for the feature_2 table
    table_data_feature_2 = pivot_feature_2.T

    # Wrap text for table headers
    table_data_feature_2.columns = [wrap_text(str(col), 5) for col in table_data_feature_2.columns]

    # Add a label for the feature_2 table
    plt.text(0.5, -0.55, f'{agg_func.capitalize()} {feature_2} by {month_col.capitalize()} and {year_col.capitalize()}', fontsize=14, fontweight='bold', ha='center', transform=ax1.transAxes)

    # Add a data point table for feature_2 below the plot
    if not table_data_feature_2.empty:
        table2 = plt.table(cellText=table_data_feature_2.values,
                           rowLabels=table_data_feature_2.index,
                           colLabels=table_data_feature_2.columns,
                           loc='bottom',
                           bbox=[0, -0.75, 1, 0.2],  # Adjusted bbox for larger table
                           cellLoc='center')

        # Format the table
        table2.auto_set_font_size(False)
        table2.set_fontsize(12)
        table2.scale(1.2, 1.5)  # Adjusted scale for bigger row height

        # Bold the fonts in the table
        for key, cell in table2.get_celld().items():
            cell.set_text_props(fontweight='bold')

    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.85, top=0.9)

    # Adjust layout for better fit
    plt.tight_layout()

    # Show the plot
    plt.show()


# Helper function to wrap text for x-axis labels and table headers
def wrap_text(text, width):
    return "\n".join(textwrap.wrap(text, width))

# Helper function to generate time series chart for different features from two dataframes
def plot_time_series(df1, df2, date_col, feature_1, feature_2, year, time_unit='week', y2_limit=None, color_1='blue', color_2='green'):
    # Filter the dataframes for the specified year
    df1 = df1[df1[date_col].dt.year == year].copy()
    df2 = df2[df2[date_col].dt.year == year].copy()

    # Add columns for the specified time unit
    if time_unit == 'month':
        df1['time_unit'] = df1[date_col].dt.to_period('M')
        df2['time_unit'] = df2[date_col].dt.to_period('M')
    elif time_unit == 'week':
        df1['time_unit'] = df1[date_col].dt.isocalendar().week
        df2['time_unit'] = df2[date_col].dt.isocalendar().week
    elif time_unit == 'day':
        df1['time_unit'] = df1[date_col].dt.date
        df2['time_unit'] = df2[date_col].dt.date
    else:
        raise ValueError("time_unit should be one of 'month', 'week', or 'day'")

    # Group the data by the specified time unit and aggregate the features
    grouped_df1 = df1.groupby('time_unit').agg({feature_1: 'sum'}).reset_index()
    grouped_df2 = df2.groupby('time_unit').agg({feature_2: 'sum'}).reset_index()

    # Merge the two dataframes on the time unit column
    merged_df = pd.merge(grouped_df1, grouped_df2, on='time_unit', how='outer').fillna(0)

    # Create figure and axis with increased height
    fig, ax1 = plt.subplots(figsize=(14, 12))

    # Plot the feature_1 time series as a bar chart on ax1
    ax1.bar(merged_df['time_unit'].astype(str), merged_df[feature_1], color=color_1, label=f'{feature_1}', alpha=0.6)

    ax1.set_xlabel(time_unit.capitalize(), fontsize=14, fontweight='bold', labelpad=20)
    ax1.set_ylabel(f'Total {feature_1}', fontsize=14, fontweight='bold', color=color_1)
    ax1.tick_params(axis='y', labelcolor=color_1)
    ax1.grid(True)

    # Rotate x-axis labels to be vertical
    ax1.set_xticks(range(len(merged_df['time_unit'])))
    ax1.set_xticklabels(merged_df['time_unit'].astype(str), rotation=90)

    # Create a twin Axes sharing the x-axis for the line chart
    ax2 = ax1.twinx()

    # Plot feature_2 as a line chart
    ax2.plot(merged_df['time_unit'].astype(str), merged_df[feature_2], color=color_2, label=f'{feature_2}', linestyle='-', marker='o')

    ax2.set_ylabel(f'Total {feature_2}', fontsize=14, fontweight='bold', color=color_2)
    ax2.tick_params(axis='y', labelcolor=color_2)

    # Optionally set the y-axis limit for feature_2
    if y2_limit:
        ax2.set_ylim(y2_limit[0], y2_limit[1])

    # Add a title to the plot
    plt.title(f'Total {feature_1} and {feature_2} by {time_unit.capitalize()} in {year}', fontsize=16, fontweight='bold')

    # Add legends to both axes
    ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 0.4), fontsize=12)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=12)

    # Prepare data for the feature_1 table
    table_data_feature_1 = merged_df[['time_unit', feature_1]].set_index('time_unit').T

    # Wrap text for table headers
    table_data_feature_1.columns = [wrap_text(str(col), 5) for col in table_data_feature_1.columns]

    # Add a label for the feature_1 table
    plt.text(0.5, -0.25, f'Total {feature_1} by {time_unit.capitalize()} in {year}', fontsize=14, fontweight='bold', ha='center', transform=ax1.transAxes)

    # Add a data point table for feature_1 below the plot
    if not table_data_feature_1.empty:
        table1 = plt.table(cellText=table_data_feature_1.values,
                          rowLabels=table_data_feature_1.index,
                          colLabels=table_data_feature_1.columns,
                          loc='bottom',
                          bbox=[0, -0.5, 1, 0.2],  # Adjusted bbox for larger table
                          cellLoc='center')

        # Format the table
        table1.auto_set_font_size(False)
        table1.set_fontsize(12)
        table1.scale(1.2, 1.5)  # Adjusted scale for bigger row height

        # Bold the fonts in the table
        for key, cell in table1.get_celld().items():
            cell.set_text_props(fontweight='bold')

    # Prepare data for the feature_2 table
    table_data_feature_2 = merged_df[['time_unit', feature_2]].set_index('time_unit').T

    # Wrap text for table headers
    table_data_feature_2.columns = [wrap_text(str(col), 5) for col in table_data_feature_2.columns]

    # Add a label for the feature_2 table
    plt.text(0.5, -0.55, f'Total {feature_2} by {time_unit.capitalize()} in {year}', fontsize=14, fontweight='bold', ha='center', transform=ax1.transAxes)

    # Add a data point table for feature_2 below the plot
    if not table_data_feature_2.empty:
        table2 = plt.table(cellText=table_data_feature_2.values,
                           rowLabels=table_data_feature_2.index,
                           colLabels=table_data_feature_2.columns,
                           loc='bottom',
                           bbox=[0, -0.75, 1, 0.2],  # Adjusted bbox for larger table
                           cellLoc='center')

        # Format the table
        table2.auto_set_font_size(False)
        table2.set_fontsize(12)
        table2.scale(1.2, 1.5)  # Adjusted scale for bigger row height

        # Bold the fonts in the table
        for key, cell in table2.get_celld().items():
            cell.set_text_props(fontweight='bold')

    plt.subplots_adjust(left=0.1, bottom=0.4, right=0.85, top=0.9)

    # Adjust layout for better fit
    plt.tight_layout()

    # Show the plot
    plt.show()


# Generic helper function to plot geomap
def plot_geomap_period(geo_df, geo_spray_df, geo_station_df, date_col, year=None, time_period='month', specific_month=None, specific_date=None, start_date=None, end_date=None, mosquito_thresholds=None):
    # Define the Coordinate Reference System (CRS)
    crs = 'EPSG:4326'  # WGS 84

    # Convert date_col to datetime
    geo_df[date_col] = pd.to_datetime(geo_df[date_col])
    geo_spray_df[date_col] = pd.to_datetime(geo_spray_df[date_col])
    
    # Filter data by year and specific time period
    if start_date and end_date:
        mask_geo_df = (geo_df[date_col] >= start_date) & (geo_df[date_col] <= end_date)
        mask_geo_spray_df = (geo_spray_df[date_col] >= start_date) & (geo_spray_df[date_col] <= end_date)
        geo_df = geo_df[mask_geo_df]
        geo_spray_df = geo_spray_df[mask_geo_spray_df]
    elif specific_date:
        geo_df = geo_df[geo_df[date_col].dt.date == specific_date]
        geo_spray_df = geo_spray_df[geo_spray_df[date_col].dt.date == specific_date]
    elif specific_month and year:
        geo_df = geo_df[(geo_df[date_col].dt.year == year) & (geo_df[date_col].dt.month == specific_month)]
        geo_spray_df = geo_spray_df[(geo_spray_df[date_col].dt.year == year) & (geo_spray_df[date_col].dt.month == specific_month)]
    elif time_period == 'month' and year:
        geo_df = geo_df[geo_df[date_col].dt.year == year]
        geo_spray_df = geo_spray_df[geo_spray_df[date_col].dt.year == year]
    elif time_period == 'week':
        geo_df = geo_df[geo_df[date_col].dt.isocalendar().week]
        geo_spray_df = geo_spray_df[geo_spray_df[date_col].dt.isocalendar().week]
    elif time_period == 'day':
        geo_df = geo_df[geo_df[date_col].dt.date]
        geo_spray_df = geo_spray_df[geo_spray_df[date_col].dt.date]
    else:
        raise ValueError("time_period should be one of 'month', 'week', 'day', or provide start_date and end_date")

    # Summing mosquito counts by location
    geo_df_sum = geo_df.groupby(['Latitude', 'Longitude']).agg({'NumMosquitos': 'sum'}).reset_index()
    geo_df_sum['geometry'] = [Point(xy) for xy in zip(geo_df_sum['Longitude'], geo_df_sum['Latitude'])]
    geo_df_sum = gpd.GeoDataFrame(geo_df_sum, crs=crs)

    # Transform GeoDataFrames to Web Mercator
    geo_df_sum = geo_df_sum.to_crs(epsg=3857)
    geo_spray_df = geo_spray_df.to_crs(epsg=3857)
    geo_station_df = geo_station_df.to_crs(epsg=3857)

    # Plotting the data
    fig, ax = plt.subplots(figsize=(11, 11))

    # Plotting spray locations as a heatmap
    spray_lat = geo_spray_df['geometry'].y
    spray_lon = geo_spray_df['geometry'].x
    sns.kdeplot(
        x=spray_lon, y=spray_lat, 
        ax=ax, cmap='YlOrRd', fill=True, 
        bw_adjust=0.5, alpha=0.5, zorder=1, cbar=True, cbar_kws={'label': 'Spray Intensity'}
    )

    # Plotting mosquito counts with solid colors based on thresholds
    mosquito_lat = geo_df_sum['geometry'].y
    mosquito_lon = geo_df_sum['geometry'].x
    mosquito_counts = geo_df_sum['NumMosquitos']
    
    thresholds = mosquito_thresholds or [100, 250, 500]
    colors = ['yellow', 'orange', 'darkorange', 'red']

    for i, threshold in enumerate(thresholds):
        if i == 0:
            mask = (mosquito_counts < threshold)
        else:
            mask = (mosquito_counts >= thresholds[i-1]) & (mosquito_counts < threshold)
        if not geo_df_sum[mask].empty:
            geo_df_sum[mask].plot(ax=ax, markersize=40, color=colors[i], marker='o', label=f'< {threshold}', zorder=2)

    if not geo_df_sum[mosquito_counts >= thresholds[-1]].empty:
        geo_df_sum[mosquito_counts >= thresholds[-1]].plot(ax=ax, markersize=40, color=colors[-1], marker='o', label=f'>= {thresholds[-1]}', zorder=2)

    # Plotting weather stations
    geo_station_df.plot(ax=ax, markersize=40, color='blue', marker='o', label='Station', zorder=3)

    # Set the bounds for the basemap based on the combined total bounds of all GeoDataFrames
    total_bounds = geo_df_sum.total_bounds
    xmin, ymin, xmax, ymax = total_bounds

    # Extend the bounds to include all points
    xmin = min(xmin, geo_station_df.total_bounds[0])
    ymin = min(ymin, geo_station_df.total_bounds[1])
    xmax = max(xmax, geo_station_df.total_bounds[2])
    ymax = max(ymax, geo_station_df.total_bounds[3])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Add the basemap using OpenStreetMap as an alternative with an appropriate zoom level
    ctx.add_basemap(ax, crs=geo_df_sum.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)

    # Adding legend and titles
    plt.legend(prop={'size': 15})
    plt.title(f'Locations of spray intensity, and weather station with Mosquito Count for {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}', fontsize=10, y=1.01)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)

plt.show()

# Helper function to plot top 8 traps by a specified feature with a stacked bar chart for each year
def plot_top_8_traps_stacked(train_withweather_df, y_feature='NumMosquitos', x_feature='Trap'):
    # Get unique years
    years = train_withweather_df['year'].unique()
    
    # Create a DataFrame to store the top 8 traps for each year
    top_8_traps_combined = pd.DataFrame()
    
    for year in years:
        # Filter data for the current year
        year_data = train_withweather_df[train_withweather_df['year'] == year]
        
        # Group by x_feature and calculate the sum for y_feature
        grouped = year_data.groupby(x_feature)[y_feature].sum().reset_index()
        
        # Sort the dataframe by y_feature and select the top 8 rows
        top_8 = grouped.nlargest(8, y_feature)
        
        # Add year column
        top_8['year'] = year
        
        # Append to the combined DataFrame
        top_8_traps_combined = pd.concat([top_8_traps_combined, top_8])
    
    # Pivot the dataframe to have years as columns
    pivot_table = top_8_traps_combined.pivot_table(index=x_feature, columns='year', values=y_feature, fill_value=0)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_table.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f'Top {x_feature} by {y_feature} (Stacked by Year)')
    plt.xticks(rotation=45)
    plt.legend(title='Year')
    plt.grid(True)
    
    # Prepare data for the table
    table_data = pivot_table.T.reset_index()
    
    # Wrap text for table headers
    table_data.columns = [wrap_text(str(col), 10) for col in table_data.columns]
    
    # Add a label for the table
    plt.text(0.5, -0.25, f'Top {x_feature} by {y_feature}', fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Add a data point table below the plot
    if not table_data.empty:
        table = plt.table(cellText=table_data.values,
                          rowLabels=table_data.index,
                          colLabels=table_data.columns,
                          loc='bottom',
                          bbox=[0, -0.5, 1, 0.2],  # Adjusted bbox for larger table
                          cellLoc='center')
        
        # Format the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)  # Adjusted scale for bigger row height
        
        # Bold the fonts in the table
        for key, cell in table.get_celld().items():
            cell.set_text_props(fontweight='bold')
    
    plt.subplots_adjust(left=0.1, bottom=0.4, right=0.85, top=0.9)
    plt.tight_layout()
    
    # Show the plot
    plt.show()


# Helper function to wrap text for table headers
def wrap_text(text, width):
    return "\n".join(textwrap.wrap(text, width))

# Helper function to get the top 8 clusters combined dataframe
def get_top_8_clusters_combined(train_weather_trapclusters, y_features):
    years = train_weather_trapclusters['year'].unique()
    top_8_clusters_combined = pd.DataFrame()

    for year in years:
        year_data = train_weather_trapclusters[train_weather_trapclusters['year'] == year]
        for y_feature in y_features:
            grouped = year_data.groupby('Trap Cluster ID')[y_feature].sum().reset_index()
            top_8 = grouped.nlargest(8, y_feature)
            top_8['year'] = year
            top_8['feature'] = y_feature
            top_8_clusters_combined = pd.concat([top_8_clusters_combined, top_8])
    
    return top_8_clusters_combined

# Helper function to plot top clusters by a specified feature with a stacked bar chart for each year
def plot_top_8_clusters_stacked(top_8_clusters_combined, y_feature):
    filtered_data = top_8_clusters_combined[top_8_clusters_combined['feature'] == y_feature]
    pivot_table = filtered_data.pivot_table(index='Trap Cluster ID', columns='year', values=y_feature, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_table.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    plt.xlabel('Trap Cluster ID')
    plt.ylabel(y_feature)
    plt.title(f'Top Clusters for each year by {y_feature} (Stacked by Year)')
    plt.xticks(rotation=45)
    plt.legend(title='Year')
    plt.grid(True)
    
    table_data = pivot_table.T.reset_index()
    table_data.columns = [wrap_text(str(col), 10) for col in table_data.columns]
    
    plt.text(0.5, -0.25, f'Top Clusters by {y_feature}', fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
    
    if not table_data.empty:
        table = plt.table(cellText=table_data.values,
                          rowLabels=table_data.index,
                          colLabels=table_data.columns,
                          loc='bottom',
                          bbox=[0, -0.5, 1, 0.2],
                          cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        for key, cell in table.get_celld().items():
            cell.set_text_props(fontweight='bold')
    
    plt.subplots_adjust(left=0.1, bottom=0.5, right=0.85, top=0.9)
    plt.tight_layout()
    plt.show()


# Define the function to get the top 25 clusters combined dataframe
def get_top_25_clusters_combined(train_weather_trapclusters):
    # Group by 'Trap Cluster ID' and calculate the sum for 'NumMosquitos' and 'WnvPresent'
    grouped = train_weather_trapclusters.groupby('Trap Cluster ID').agg({
        'NumMosquitos': 'sum',
        'WnvPresent': 'sum'
    }).reset_index()
    
    # Sort the dataframe by 'NumMosquitos' and select the top 25 rows
    top_25_clusters_combined = grouped.nlargest(25, 'NumMosquitos')
    
    # Merge to get the cluster center latitude and longitude
    top_25_clusters_combined = top_25_clusters_combined.merge(
        train_weather_trapclusters[['Trap Cluster ID', 'trapclust_lat', 'trapclust_long']].drop_duplicates(),
        on='Trap Cluster ID',
        how='left'
    )
    
    # Rename columns for clarity
    top_25_clusters_combined = top_25_clusters_combined.rename(
        columns={'trapclust_lat': 'cluster_latitude', 'trapclust_long': 'cluster_longitude'}
    )
    
    return top_25_clusters_combined


# Define the function to plot geomap for a period and include top 25 trap clusters
def plot_geomap_clusters_period(geo_df, geo_station_df, top_25_clusters_combined, date_col, year=None, time_period='month', specific_month=None, specific_date=None, start_date=None, end_date=None, mosquito_thresholds=None):
    # Define the Coordinate Reference System (CRS)
    crs = 'EPSG:4326'  # WGS 84

    # Convert date_col to datetime
    geo_df[date_col] = pd.to_datetime(geo_df[date_col])
    
    # Filter data by year and specific time period
    if start_date and end_date:
        mask_geo_df = (geo_df[date_col] >= start_date) & (geo_df[date_col] <= end_date)
        geo_df = geo_df[mask_geo_df]
    elif specific_date:
        geo_df = geo_df[geo_df[date_col].dt.date == specific_date]
    elif specific_month and year:
        geo_df = geo_df[(geo_df[date_col].dt.year == year) & (geo_df[date_col].dt.month == specific_month)]
    elif time_period == 'month' and year:
        geo_df = geo_df[geo_df[date_col].dt.year == year]
    elif time_period == 'week':
        geo_df = geo_df[geo_df[date_col].dt.isocalendar().week == year]
    elif time_period == 'day':
        geo_df = geo_df[geo_df[date_col].dt.date == year]
    else:
        raise ValueError("time_period should be one of 'month', 'week', 'day', or provide start_date and end_date")

    # Filter geo_df to include only the top 25 clusters for the specified period
    geo_df = geo_df[geo_df['Trap Cluster ID'].isin(top_25_clusters_combined['Trap Cluster ID'].unique())]
    
    # Summing mosquito counts by location
    geo_df_sum = geo_df.groupby(['trapclust_lat', 'trapclust_long']).agg({'NumMosquitos': 'sum'}).reset_index()
    geo_df_sum['geometry'] = [Point(xy) for xy in zip(geo_df_sum['trapclust_long'], geo_df_sum['trapclust_lat'])]
    geo_df_sum = gpd.GeoDataFrame(geo_df_sum, crs=crs)

    # Transform GeoDataFrames to Web Mercator
    geo_df_sum = geo_df_sum.to_crs(epsg=3857)
    geo_station_df = geo_station_df.to_crs(epsg=3857)

    # Plotting the data
    fig, ax = plt.subplots(figsize=(11, 11))

    # Plotting mosquito counts with solid colors based on thresholds
    mosquito_lat = geo_df_sum['geometry'].y
    mosquito_lon = geo_df_sum['geometry'].x
    mosquito_counts = geo_df_sum['NumMosquitos']
    
    thresholds = mosquito_thresholds or [100, 250, 500]
    colors = ['yellow', 'orange', 'darkorange', 'red']

    for i, threshold in enumerate(thresholds):
        if i == 0:
            mask = (mosquito_counts < threshold)
        else:
            mask = (mosquito_counts >= thresholds[i-1]) & (mosquito_counts < threshold)
        if not geo_df_sum[mask].empty:
            geo_df_sum[mask].plot(ax=ax, markersize=120, color=colors[i], marker='o', label=f'< {threshold}', zorder=2)

    if not geo_df_sum[mosquito_counts >= thresholds[-1]].empty:
        geo_df_sum[mosquito_counts >= thresholds[-1]].plot(ax=ax, markersize=120, color=colors[-1], marker='o', label=f'>= {thresholds[-1]}', zorder=2)

    # Plotting weather stations
    geo_station_df.plot(ax=ax, markersize=120, color='blue', marker='o', label='Station', zorder=3)

    # Set the bounds for the basemap based on the combined total bounds of all GeoDataFrames
    total_bounds = geo_df_sum.total_bounds
    xmin, ymin, xmax, ymax = total_bounds

    # Extend the bounds to include all points
    xmin = min(xmin, geo_station_df.total_bounds[0])
    ymin = min(ymin, geo_station_df.total_bounds[1])
    xmax = max(xmax, geo_station_df.total_bounds[2])
    ymax = max(ymax, geo_station_df.total_bounds[3])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Add the basemap using OpenStreetMap as an alternative with an appropriate zoom level
    ctx.add_basemap(ax, crs=geo_df_sum.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)

    # Adding legend and titles
    plt.legend(prop={'size': 15})
    if start_date and end_date:
        plt.title(f'Top 25 Trap Clusters with Mosquito Count for {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}', fontsize=10, y=1.01)
    elif specific_date:
        plt.title(f'Top 25 Trap Clusters with Mosquito Count for {specific_date.strftime("%Y-%m-%d")}', fontsize=10, y=1.01)
    elif specific_month and year:
        plt.title(f'Top 25 Trap Clusters with Mosquito Count for {year}-{specific_month:02d}', fontsize=10, y=1.01)
    elif year:
        plt.title(f'Top 25 Trap Clusters with Mosquito Count for {year}', fontsize=10, y=1.01)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)

    plt.show()


#functions for modelling

def gridsearch_model_eval(model, param_grid, X_train, y_train, X_test, y_test, evaluation_results, gridsearch=True):
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    model_name = model.__class__.__name__
    
    if gridsearch:
        adjusted_param_grid = {f'classifier__{key}': value for key, value in param_grid.items()}

        grid_search = GridSearchCV(estimator=pipeline,
                                   param_grid=adjusted_param_grid,
                                   scoring='roc_auc',
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=1)

        grid_search.fit(X_train, y_train)

        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", round(grid_search.best_score_, 3))

        best_model = grid_search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        best_model = pipeline

        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        print(f"Cross-validation ROC_AUC scores: {cv_scores}")
        print(f"Mean CV ROC_AUC score: {cv_scores.mean()}")

    y_train_pred = best_model.predict(X_train)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = best_model.predict(X_test)

    train_accuracy = round(accuracy_score(y_train, y_train_pred), 3)
    val_roc_auc = round(roc_auc_score(y_test, y_test_prob), 3)
    val_accuracy = round(accuracy_score(y_test, y_test_pred), 3)
    val_conf_matrix = confusion_matrix(y_test, y_test_pred)
    val_class_report = classification_report(y_test, y_test_pred)

    print("ROC_AUC Score:", val_roc_auc)
    #print("Validation Accuracy:", val_accuracy)
    print("Validation Confusion Matrix:\n", val_conf_matrix)
    print("Validation Classification Report:\n", val_class_report)

    if gridsearch:
        evaluation_results[model_name] = {
            'Best Model': best_model,
            'Best Parameters': grid_search.best_params_,
            'Best Score': round(grid_search.best_score_, 3),
            'Training Accuracy': train_accuracy,
            'Validation Accuracy': val_accuracy,
            'ROC_AUC Score': val_roc_auc
        }
        model_filename = f"{model_name}_tuned"
    else:
        evaluation_results[f"{model_name}_baseline"] = {
            'Best Model': best_model,
            'Cross-validation Scores': cv_scores,
            'Mean CV Score': cv_scores.mean(),
            'Training Accuracy': train_accuracy,
            'Validation Accuracy': val_accuracy,
            'ROC_AUC Score': val_roc_auc
        }
        model_filename = f"{model_name}_baseline"

    cm = val_conf_matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap='YlOrRd', fmt='g',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {val_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    model_path = f"model/{model_filename}.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

    # Select a random instance index for SHAP analysis
    instance_index = np.random.randint(0, X_train.shape[0])
    
    # Return the best model, preprocessor, and instance_index for SHAP analysis
    return best_model, preprocessor, instance_index

#ROC-AUC

def plot_roc_auc_and_accuracy(evaluation_results, X_test, y_test):
    """
    Plot ROC-AUC curves and a horizontal bar graph for accuracy scores for all models in the evaluation_results dictionary.

    Parameters:
    evaluation_results (dict): A dictionary containing evaluation results of different models.
    X_test (DataFrame or array-like): The test data.
    y_test (array-like): The true labels for the test data.
    """
    def plot_roc_auc(evaluation_results, X_test, y_test):
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and a set of subplots with specified size

        for model_name, results in evaluation_results.items():
            best_model = results['Best Model']
            y_test_prob = best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            roc_auc = results['ROC_AUC Score']

            ax.plot(fpr, tpr, lw=2, label=f'{model_name} (area = {roc_auc:.2f})')

        ax.legend(loc="lower right")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        plt.show()

    plot_roc_auc(evaluation_results, X_test, y_test)

#Shap plot
import shap
import matplotlib.pyplot as plt
import pandas as pd

def plot_top_shap(best_model, X_train, top_n=5):
    """
    Plot the top N features based on SHAP values in reverse order.

    Parameters:
    best_model (Pipeline): The trained pipeline model.
    X_train (DataFrame or array-like): The training data.
    top_n (int): The number of top features to plot. Default is 5.
    """
    # Extract the fitted preprocessor from the best model
    fitted_preprocessor = best_model.named_steps['preprocessor']
    # Preprocess the training data
    X_train_preprocessed = fitted_preprocessor.transform(X_train)
    # Initialize the SHAP explainer with the best model
    explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
    shap_values = explainer.shap_values(X_train_preprocessed)
    feature_names = fitted_preprocessor.get_feature_names_out()
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    feature_importance = shap_df.abs().mean().sort_values(ascending=False)
    top_features = feature_importance.head(top_n).sort_values(ascending=True)

    # Plotting top N features in reverse order
    plt.figure(figsize=(5, 4))
    top_features.plot(kind='barh')
    plt.title(f'Top {top_n} Features')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Features')
    plt.show()

# Function to plot shap_features
def plot_top_shap_features(shap_df, top_n=5):
    """
    Plot the top SHAP features.

    Parameters:
    shap_df (pd.DataFrame): DataFrame containing SHAP values with feature names.
    top_n (int): Number of top features to plot.

    Returns:
    None
    """
    # Calculate the mean absolute SHAP values
    mean_abs_shap_values = shap_df.abs().mean()
    
    # Get the top N features
    top_n_features = mean_abs_shap_values.sort_values(ascending=False).index[:top_n]
    
    # Extract SHAP values for the top features
    shap_values_top_n = shap_df[top_n_features].values

    # Plotting Top N features
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values_top_n, features=shap_df[top_n_features], feature_names=top_n_features, cmap=plt.cm.PiYG, show=False)
    ax.patch.set_alpha(0)  # Make the axes patch (background) transparent
    fig.patch.set_alpha(0)  # Make the figure background transparent
    for spine in ax.spines.values():
        spine.set_visible(False)  # Remove the spines
    ax.grid(True, linestyle='--', alpha=0.7, color='gray')  # Customize the gridlines
    plt.title('Top {} SHAP Features'.format(top_n), backgroundcolor='none')
    plt.show()

# Plotting ROC_AUC of all trained models
def plot_roc_auc_scores(evaluation_results):
    """
    Plot the ROC AUC scores of classifiers.

    Parameters:
    evaluation_results (dict): Dictionary containing classifier names and their evaluation details.

    Returns:
    None
    """
    # Extract ROC AUC scores from evaluation_results
    roc_auc_scores = {clf: details['ROC_AUC Score'] for clf, details in evaluation_results.items()}

    # Exclude AdaBoostClassifier
    filtered_scores = {clf: score for clf, score in roc_auc_scores.items() if clf != 'AdaBoostClassifier'}

    # Rename XGBClassifier to XGBClassifier_tuned
    filtered_scores['XGBClassifier_tuned'] = filtered_scores.pop('XGBClassifier')

    # Remove _baseline from all except RandomForestClassifier
    filtered_scores = {clf.replace('_baseline', '') if clf != 'RandomForestClassifier_baseline' else clf: score 
                       for clf, score in filtered_scores.items()}

    # Sorting the scores in descending order
    sorted_filtered_scores = dict(sorted(filtered_scores.items(), key=lambda item: item[1], reverse=False))

    # Extracting sorted classifier names and their respective ROC AUC scores
    classifiers_sorted_filtered = list(sorted_filtered_scores.keys())
    roc_auc_scores_sorted_filtered = list(sorted_filtered_scores.values())

    # Determine colors for each bar
    colors = ['lightgreen' if clf == 'XGBClassifier_tuned' else 'black' if clf == 'RandomForestClassifier_baseline' else 'lightblue'
              for clf in classifiers_sorted_filtered]

    # Creating the horizontal bar plot with sorted scores
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
    bars = ax.barh(classifiers_sorted_filtered, roc_auc_scores_sorted_filtered, color=colors)
    ax.set_xlabel('ROC AUC Score')
    ax.set_title('ROC AUC Scores of Classifiers')
    ax.set_xlim(0.6, 0.9)

    # Customize the gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.7, color='gray')

    # Make the axes patch (background) transparent
    ax.patch.set_alpha(0)

    # Set the figure background to be transparent
    fig.patch.set_alpha(0)

    plt.show()


# Extract additional datetime features
def extract_date_features(df):
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['IsWeekend'] = df['DayOfWeek'] >= 5
    return df


# Function to extract ZIP code from address
def extract_zipcode(address, manual_zip_mapping):
    if address in manual_zip_mapping:
        return 'z'+ manual_zip_mapping[address]
    match = re.search(r'\b\d{5}\b', address)
    if match:
        return 'z' + match.group(0)
    return 'N/A'


def convert_columns_to_numeric(df):
    for column in df.columns:
        if column.lower() == 'date':
            try:
                df[column] = pd.to_datetime(df[column])
                df[column] = df[column].astype('int64') // 10**9 // 86400  # Convert to days since 1970-01-01
                print(f"Converted {column} to numeric (days since 1970-01-01).")
            except ValueError:
                print(f"Could not convert {column} to datetime. Skipping...")
            continue
        if column.lower() == 'zip':
            print(f"Skipping conversion for {column} column.")
            continue
        if column.lower() == 'block':
            df[column] = 'BLOCK ' + df[column].astype(str)
            print(f"Added 'BLOCK' to all entries in {column} column.")
            continue
        if df[column].dtype == 'bool':
            df[column] = df[column].astype(int)
            print(f"Converted {column} to numeric (boolean to int).")
            continue
        try:
            df[column] = pd.to_numeric(df[column])
            print(f"Converted {column} to numeric.")
        except ValueError:
            print(f"Could not convert {column} to numeric. Skipping...")
    return df



# Create a DataFrame to list out clusters and unique trap IDs
def get_clusters_and_traps(train_withweather_df):
    # Group by 'Trap Cluster ID' and collect unique trap IDs
    cluster_traps = train_withweather_df.groupby('Trap Cluster')['Trap'].unique().reset_index()
    
    # Rename columns for clarity
    cluster_traps = cluster_traps.rename(columns={'Trap Cluster': 'Cluster', 'Trap': 'Trap IDs'})
    
    # Sort by Cluster for better readability
    cluster_traps = cluster_traps.sort_values(by='Cluster').reset_index(drop=True)
    
    return cluster_traps
