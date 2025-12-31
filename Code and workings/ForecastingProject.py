import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from io import BytesIO
from datetime import datetime
from geopy.distance import geodesic
from math import sqrt
from pathlib import Path

# Try to import PyMuPDF (fitz). If unavailable, skip PDF generation later and warn the user.
try:
    import fitz
except Exception as e:
    fitz = None
    print("Warning: PyMuPDF (fitz) is not installed or failed to import. PDF generation will be skipped. To enable PDF output, install PyMuPDF: pip install pymupdf")

# Base dir for data files (script directory)
BASE_DIR = Path(__file__).resolve().parent

# Load datasets (use absolute paths based on script location)
att_df = pd.read_csv(BASE_DIR / "AttendancesMidlands.csv", parse_dates=["Date"])
obs_df = pd.read_csv(BASE_DIR / "Observation2025.csv", parse_dates=["DateTime"])
hist_temp_df = pd.read_csv(BASE_DIR / "MetOfficeObservation.csv", parse_dates=["DateTime"])

# Preprocess attendance data
att_df = att_df.rename(columns={"Filter1": "ICBS"})
att_df = att_df[att_df["ICBS"].notna()]
att_df["Week"] = att_df["Date"].dt.isocalendar().week
att_df["Month"] = att_df["Date"].dt.month
att_df["Year"] = att_df["Date"].dt.year
att_df["DayOfWeek"] = att_df["Date"].dt.dayofweek

# Preprocess observation data
obs_df = obs_df.rename(columns={"DateTime": "Date"})
hist_temp_df = hist_temp_df.rename(columns={"DateTime": "Date"})


# Correct ICBS name mismatch
att_df["ICBS"] = att_df["ICBS"].replace({
    "SHROPSHIRE, TELFORD AND WREKIN": "SHROPSHIRE AND TELFORD AND WREKIN"
})


# Define ICBS coordinates
icbs_coords = {
    "BIRMINGHAM AND SOLIHULL": (52.4862, -1.8904),
    "BLACK COUNTRY": (52.5386, -2.0175),
    "COVENTRY AND WARWICKSHIRE": (52.4081, -1.5106),
    "HEREFORDSHIRE AND WORCESTERSHIRE": (52.1951, -2.2200),
    "SHROPSHIRE AND TELFORD AND WREKIN": (52.7073, -2.7481),
    "STAFFORDSHIRE AND STOKE-ON-TRENT": (52.8050, -2.1164),
    "DERBY AND DERBYSHIRE": (52.9225, -1.4746),
    "LEICESTER, LEICESTERSHIRE AND RUTLAND": (52.6369, -1.1398),
    "LINCOLNSHIRE": (53.2344, -0.5380),
    "NORTHAMPTONSHIRE": (52.2405, -0.9027),
    "NOTTINGHAM AND NOTTINGHAMSHIRE": (52.9548, -1.1581)
}

# Get unique weather stations with coordinates
weather_sites = obs_df[["WeatherSiteName", "SiteLatitude", "SiteLongitude"]].dropna().drop_duplicates()

# Map ICBS to nearest weather station
icbs_to_station = {}
for icbs, icbs_coord in icbs_coords.items():
    min_dist = float("inf")
    nearest_station = None
    for _, row in weather_sites.iterrows():
        station_coord = (row["SiteLatitude"], row["SiteLongitude"])
        dist = geodesic(icbs_coord, station_coord).kilometers
        if dist < min_dist:
            min_dist = dist
            nearest_station = row["WeatherSiteName"]
    icbs_to_station[icbs] = nearest_station

# Prepare training data for temperature model
hist_temp_df["Month"] = hist_temp_df["Date"].dt.month
hist_temp_df["Day"] = hist_temp_df["Date"].dt.day
hist_temp_df["Year"] = hist_temp_df["Date"].dt.year
hist_temp_df = hist_temp_df[hist_temp_df["Month"].between(7, 12)]

# Train temperature model
temp_model = HistGradientBoostingRegressor()
temp_features = ["Month", "Day", "Year", "SiteLatitude", "SiteLongitude", "Elevation"]
temp_df = hist_temp_df[temp_features + ["Temperature"]].dropna()
X_temp = temp_df[temp_features]
y_temp = temp_df["Temperature"]
temp_model.fit(X_temp, y_temp)

# Evaluate temperature model
y_pred_temp = temp_model.predict(X_temp)
rmse_temp = sqrt(mean_squared_error(y_temp, y_pred_temp))
mae_temp = mean_absolute_error(y_temp, y_pred_temp)
r2_temp = r2_score(y_temp, y_pred_temp)
print("Temperature RMSE:", rmse_temp)
print("Temperature MAE:", mae_temp)
print("Temperature R²:", r2_temp)

# Evaluate temperature model
y_pred_temp = temp_model.predict(X_temp)
rmse_temp = sqrt(mean_squared_error(y_temp, y_pred_temp))
r2_temp = r2_score(y_temp, y_pred_temp)

# Forecast dates
forecast_dates = pd.date_range(start="2025-07-01", end="2025-12-31", freq="W")
forecast_df = pd.DataFrame({"Date": forecast_dates})
forecast_df["Month"] = forecast_df["Date"].dt.month
forecast_df["Day"] = forecast_df["Date"].dt.day
forecast_df["Year"] = forecast_df["Date"].dt.year

# Predict temperature for each ICBS
temp_forecasts = []
for icbs, station in icbs_to_station.items():
    station_info = weather_sites[weather_sites["WeatherSiteName"] == station]
    if station_info.empty:
        continue
    lat = station_info["SiteLatitude"].values[0]
    lon = station_info["SiteLongitude"].values[0]
    elev = hist_temp_df[hist_temp_df["WeatherSiteName"] == station]["Elevation"].dropna()
    elev = elev.values[0] if not elev.empty else 100.0
    df = forecast_df.copy()
    df["SiteLatitude"] = lat
    df["SiteLongitude"] = lon
    df["Elevation"] = elev
    df["ICBS"] = icbs
    df["PredictedTemp"] = temp_model.predict(df[temp_features])
    temp_forecasts.append(df[["Date", "ICBS", "PredictedTemp"]])

temp_forecast_all = pd.concat(temp_forecasts)

# Merge temperature forecast with attendance data
att_df["WeatherSiteName"] = att_df["ICBS"].map(icbs_to_station)
merged_df = pd.merge(att_df, obs_df[["WeatherSiteName", "Date", "Temperature"]],
                     on=["WeatherSiteName", "Date"], how="left")

# Add public holidays (UK Bank Holidays 2025)
uk_holidays_2025 = pd.to_datetime([
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05", "2025-05-26",
    "2025-08-25", "2025-12-25", "2025-12-26"
])
merged_df["IsHoliday"] = merged_df["Date"].isin(uk_holidays_2025).astype(int)

# Format the holidays as strings with weekday
holiday_strings = [d.strftime("%a, %b %d") for d in uk_holidays_2025]


# Prepare training data for attendance model
train_df = merged_df[(merged_df["Date"] < "2025-07-01")]
train_df["DayOfWeek"] = train_df["Date"].dt.dayofweek
train_df["IsHoliday"] = train_df["Date"].isin(uk_holidays_2025).astype(int)
train_df = train_df[["ICBS", "Week", "Month", "DayOfWeek", "IsHoliday", "Temperature", "Value"]].dropna(subset=["Value"])
train_df = pd.get_dummies(train_df, columns=["ICBS"], drop_first=True)
imputer = SimpleImputer(strategy="mean")
X_train = train_df.drop(columns=["Value"])
X_train_imputed = imputer.fit_transform(X_train)
y_train = train_df["Value"]
att_model = HistGradientBoostingRegressor()
att_model.fit(X_train_imputed, y_train)

# Forecast attendance using predicted temperature
forecast_df = pd.DataFrame({"Date": forecast_dates})
forecast_df["Week"] = forecast_df["Date"].dt.isocalendar().week
forecast_df["Month"] = forecast_df["Date"].dt.month
forecast_df["DayOfWeek"] = forecast_df["Date"].dt.dayofweek
forecast_df["IsHoliday"] = forecast_df["Date"].isin(uk_holidays_2025).astype(int)

forecast_results = []
for icbs in att_df["ICBS"].unique():
    df_temp = temp_forecast_all[temp_forecast_all["ICBS"] == icbs]
    if df_temp.empty:
        continue
    df = forecast_df.merge(df_temp[["Date", "PredictedTemp"]], on="Date", how="left")
    df["ICBS"] = icbs
    df = pd.get_dummies(df, columns=["ICBS"])
    for col in X_train.columns:
        if col not in df.columns:
            df[col] = 0
    df = df[X_train.columns]
    df_imputed = imputer.transform(df)
    df["PredictedAttendance"] = att_model.predict(df_imputed)
    df["ICBS"] = icbs
    df["Date"] = forecast_dates.values
    forecast_results.append(df[["Date", "ICBS", "PredictedAttendance"]])

forecast_att_all = pd.concat(forecast_results)

# Merge temperature and attendance forecasts
combined_forecast = pd.merge(forecast_att_all, temp_forecast_all, on=["Date", "ICBS"], how="left")

# Function to annotate holidays at the top
def annotate_holidays(ax, holidays):
    for holiday in sorted(holidays):
        if holiday.year == 2025:
            ax.axvline(holiday, color='red', linestyle=':', alpha=0.6, zorder=5)
            label = holiday.strftime('%b %d')
            x_offset = 0
            if holiday.strftime('%m-%d') == '12-25':
                x_offset = -3
            elif holiday.strftime('%m-%d') == '12-26':
                x_offset = 5
            elif holiday.strftime('%m-%d') == '04-18':
                x_offset = -3
            elif holiday.strftime('%m-%d') == '04-21':
                x_offset = 5
            y_top = ax.get_ylim()[1] + (ax.get_ylim()[1] * 0.05)
            ax.annotate(label,
                        xy=(holiday, y_top),
                        xytext=(holiday + pd.Timedelta(days=x_offset), y_top),
                        textcoords='data',
                        rotation=75,
                        color='red',
                        fontsize=8,
                        ha='center',
                        arrowprops=dict(arrowstyle='-', color='red', lw=0.5),
                        zorder=6)

# Generate PDF with plots (only if PyMuPDF is available)
if fitz is not None:
    pdf_doc = fitz.open()
    for icbs in combined_forecast["ICBS"].unique():
        df_forecast = combined_forecast[combined_forecast["ICBS"] == icbs]
        df_actual_att = att_df[(att_df["ICBS"] == icbs) & (att_df["Date"].dt.year == 2025) & (att_df["Date"] < "2025-07-01")]
        df_actual_temp = merged_df[(merged_df["ICBS"] == icbs) & (merged_df["Date"].dt.year == 2025) & (merged_df["Date"] < "2025-07-01")]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df_actual_att["Date"], df_actual_att["Value"], 'go', label="Actual Attendance")
        ax1.plot(df_forecast["Date"], df_forecast["PredictedAttendance"], 'g^-', label="Predicted Attendance")
        ax1.set_ylabel("Attendance (visits)", color='green')
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.set_xlabel("Date")
        ax1.set_title(f"{icbs} – Attendance & Temperature (2025)\nActual (Jan–Jun) vs. ML Forecast (Jul–Dec)")
        ax1.grid(True)
        ax1.axvline(pd.Timestamp("2025-07-01"), color='gray', linestyle='--', label="Forecast Start")

        # Annotate holidays
        annotate_holidays(ax1, uk_holidays_2025)

        ax2 = ax1.twinx()
        ax2.plot(df_actual_temp["Date"], df_actual_temp["Temperature"].rolling(3).mean(), 'bo', label="Actual Temperature (smoothed)")
        ax2.plot(df_forecast["Date"], df_forecast["PredictedTemp"], 'b^-', label="Predicted Temperature")
        ax2.set_ylabel("Temperature (°C)", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   ncol=2, frameon=True, fontsize='small', title='Legend')

        fig.tight_layout()
        img_buf = BytesIO()
        plt.savefig(img_buf, format="png", bbox_inches="tight")
        img_buf.seek(0)
        page = pdf_doc.new_page(width=595, height=842)
        rect = fitz.Rect(0, 0, 595, 842)
        page.insert_image(rect, stream=img_buf.read())
        plt.close(fig)


    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error, r2_score    

    y_pred_att = att_model.predict(X_train_imputed)
    print("Attendance RMSE:", sqrt(mean_squared_error(y_train, y_pred_att)))
    print("Attendance MAE:", mean_absolute_error(y_train, y_pred_att))
    print("Attendance R²:", r2_score(y_train, y_pred_att))


    y_pred_temp = temp_model.predict(X_temp)
    print("Temperature RMSE:", sqrt(mean_squared_error(y_temp, y_pred_temp)))
    print("Temperature MAE:", mean_absolute_error(y_temp, y_pred_temp))
    print("Temperature R²:", r2_score(y_temp, y_pred_temp))


    # Save PDF
    pdf_path = "Enhanced_ICBS_Forecast_2025_Seasonality_Holidays.pdf"
    pdf_doc.save(pdf_path)
    pdf_doc.close()


    # Load the original PDF
    input_pdf_path = pdf_path
    output_pdf_path = "Enhanced_ICBS_Forecast_2025_Seasonality_Holidays_WithHolidayTable.pdf"

    doc = fitz.open(input_pdf_path)

    # Define layout parameters
    font_size = 8
    line_height = 10
    start_y = 575  # position below the legend
    start_x = 260

    # Add holiday table to each page
    for page in doc:
        y = start_y
        page.insert_text((start_x, y), "UK Bank Holidays 2025:", fontsize=font_size + 1, fontname="helv", fill=(0, 0, 0))
        y += line_height
        for holiday in holiday_strings:
            page.insert_text((start_x, y), f"- {holiday}", fontsize=font_size, fontname="helv", fill=(0, 0, 0))
            y += line_height

    # Save the updated PDF
    doc.save(output_pdf_path)
    doc.close()

    print(f"✅ Updated PDF saved as: {output_pdf_path}")


    print(f"✅ Forecast PDF saved to {pdf_path}")
else:
    print("Skipping PDF generation and holiday-table enhancement because PyMuPDF is not available. Install it with: pip install pymupdf")




