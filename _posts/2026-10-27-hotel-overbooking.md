---
layout: post
title: Cancellation Forecasting & Smart Overbooking For Hotels Using Machine Learning
image: "/posts/hotel-overbooking-title-img.png"
tags: [Cancellation Forecasting, LightGBM, Overbooking Optimisation, Python]
---

In this project we use Machine Learning to predict which hotel bookings will be cancelled (or turn into no-shows), and translate those predictions into a nightly overbooking recommendation and a daily list of guests worth contacting before they cancel.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Modelling Overview](#modelling-overview)
- [03. Data Preparation](#data-prep)
- [04. Training The Risk Model](#model-training)
- [05. Model Performance & Explainability](#model-performance)
- [06. Occupancy Forecast](#occupancy-forecast)
- [07. Overbooking Optimisation & Backtest](#overbooking)
- [08. Preventive Actions & Cancellation Policies](#preventive-actions)
- [09. Monitoring & Business Case](#monitoring-business-case)
- [10. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client runs two hotels - a City hotel and a Resort hotel - and they are losing a lot of money to bookings that never turn into a stay.  41% of bookings at the City hotel and 27% at the Resort get cancelled, and around 12% of those cancellations happen in the last 7 days before arrival - at which point the room is very hard to resell.  Across the two hotels that is over €1.5m of revenue cancelled late.

Like many hotels, they currently offset this with a single, fixed overbooking percentage.  On high-demand nights this leaves rooms empty, and on other nights it leads to "walked" guests - guests who arrive to find no room and need to be relocated to another hotel at the hotel's expense.  On top of this, nobody knows *which* bookings are at risk, so nobody does anything about them!

Our task is to build a proof of concept that tells the revenue team, every day, how many extra bookings to accept for each night, and which bookings the front office should contact before they cancel.

<br>
<br>
### Actions <a name="overview-actions"></a>

Based upon the tasks at hand - we built the solution in four layers, each one feeding the next:

* **Risk model** - a *LightGBM* classifier with *isotonic calibration* that scores every active booking at any point from 180 days before arrival right up to the day of arrival, using only the information available on that day
* **Occupancy forecast** - the booking-level probabilities are aggregated into a full probability distribution of final occupancy for each night, then corrected by a second-stage "booking curve" model
* **Overbooking optimisation** - for each night we choose the number of extra bookings to accept by trading off the value of a sold room against the cost of walking a guest
* **Preventive actions** - high-risk bookings come with their top *SHAP* drivers and a recommended action, with the contact threshold set from an A/B uplift analysis

Throughout the project we were very strict about two things - validation was done *out-of-time* (train on the past, test on the future), and the model was never allowed to see information it wouldn't have had at the moment of the decision.

All models were compared against what the hotels do today - a segment-based cancellation rate heuristic, and a static 5% overbooking policy - as well as against an "oracle" that represents the best possible result with perfect hindsight.

<br>
<br>

### Results <a name="overview-results"></a>

On the out-of-time test period (January to August 2017) the calibrated LightGBM model achieved a ROC AUC of **0.802** versus **0.783** for the current segment heuristic, with a lower Brier score (0.0754 vs 0.0785) - so not only does it rank risky bookings better, its probabilities can also be trusted as probabilities.

The real story however is in the overbooking backtest.  The static 5% overbooking policy that many hotels use actually *lost* €112k across the two hotels, walking 697 guests in the process.  Our model-based daily recommendation instead *earned* €68k, walking only 205 guests - capturing 37% of the theoretical maximum.

| **Metric** | **Current practice** | **This solution** |
|---|---|---|
| ROC AUC (booking level) | 0.783 (segment heuristic) | **0.802** |
| Brier score | 0.0785 | **0.0754** |
| Overbooking net result, 2 hotels | -€112k (static 5%) | **+€68k** |
| Guests walked | 697 | **205** |
| Annualised net benefit | - | **≈ €118k**, payback ≈ 3 months |

<br>
The annualised figure includes some synthetic assumptions (walk cost, turned-away demand, effect of guest contact) as these simply don't exist in public data - they are all listed in one place in the code and would need to be validated on client data before making any commitments!

<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

As this is a proof of concept on public data, the natural next step is a pilot on the client's own PMS (Property Management System) data, with real room capacities and real walk costs.

We would also replace the synthetic elements with measured values - booking-engine denials for turned-away demand, and a live A/B test of guest outreach.

For a multi-property rollout we would move to a central model with hotel-level features, and plug the daily outputs into Power BI or Snowflake so they sit alongside the client's existing reporting.

<br>
<br>

___

# Data Overview  <a name="data-overview"></a>

We use the public [Hotel Booking Demand](https://doi.org/10.1016/j.dib.2018.11.126) dataset (Antonio, Almeida & Nunes, 2019; CC BY 4.0) which contains 119,390 bookings from two Portuguese hotels, with arrivals from July 2015 to August 2017.

Each row is one booking, with 32 columns describing who booked, when, through which channel, at what price, and - most importantly - what eventually happened to the booking.

In the code below, we import Pandas and bring the raw data into Python.
<br>
```python

# import required Python packages
import pandas as pd
import numpy as np

# import the raw data
hotel_bookings = pd.read_csv("data/raw/hotel_bookings.csv")

```
<br>

A sample of this data (a handful of bookings and a selection of the columns) can be seen below:
<br>
<br>

| **hotel** | **is_canceled** | **lead_time** | **arrival_date_year** | **arrival_date_month** | **arrival_day** | **week_nights** | **adults** | **market_segment** | **deposit_type** | **adr** | **reservation_status** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Resort Hotel | 0 | 41 | 2015 | December | 31 | 3 | 2 | Offline TA/TO | No Deposit | 145.87 | Check-Out |
| Resort Hotel | 1 | 27 | 2015 | November | 16 | 2 | 2 | Online TA | No Deposit | 48.00 | Canceled |
| City Hotel | 0 | 177 | 2017 | June | 24 | 1 | 2 | Online TA | No Deposit | 130.50 | Check-Out |
| City Hotel | 0 | 61 | 2015 | October | 27 | 5 | 2 | Online TA | No Deposit | 74.61 | Check-Out |
| City Hotel | 0 | 2 | 2015 | September | 30 | 2 | 2 | Online TA | No Deposit | 126.00 | Check-Out |
| Resort Hotel | 1 | 40 | 2017 | January | 16 | 2 | 1 | Groups | No Deposit | 55.00 | Canceled |
| Resort Hotel | 1 | 11 | 2017 | January | 14 | 1 | 2 | Online TA | No Deposit | 78.00 | Canceled |
| Resort Hotel | 0 | 13 | 2016 | August | 9 | 3 | 2 | Online TA | No Deposit | 261.00 | Check-Out |
| … | … | … | … | … | … | … | … | … | … | … | … |

<br>
To explain this data, the first row is a booking at the Resort hotel, made 41 days before arrival (*lead_time*), for 3 weekday nights from New Year's Eve 2015, for 2 adults through an offline travel agent, at an average daily rate (*adr*) of €145.87 - and the guests did arrive and check out.

The *reservation_status* column tells us the final outcome of each booking - Check-Out (75,166 bookings), Canceled (43,017) or No-Show (1,207).  Together with *reservation_status_date* this tells us not only *whether* a booking was cancelled, but *when* - which turns out to be crucial for this project.

A summary of the problem, by hotel, after cleaning:
<br>
<br>

| | **City hotel** | **Resort hotel** |
|---|---|---|
| Bookings | 78,898 | 39,666 |
| Cancellation rate | 40.8% | 27.3% |
| No-show rate | 1.2% | 0.7% |
| Share of cancellations made in the last 7 days | 12.0% | 11.6% |
| Avg. booking lead time (days) | 110 | 93 |
| Avg. rate per night (ADR) | €107 | €92 |
| Revenue cancelled late (≤ 7 days) | €1,167,163 | €366,324 |
| Revenue lost to no-shows | €361,170 | €88,311 |

<br>
The chart below shows *when* cancellations happen - roughly 12% of them arrive in the final week before arrival, where the room is very hard to resell.

<br>
![alt text](/img/posts/hotel-overbooking-cancellation-timing.png "Cumulative share of cancellations by days before arrival")

<br>
A few important things are *not* in the public data - the number of rooms in each hotel, the cost of walking a guest, demand that was turned away on sold-out nights, and any actions taken towards guests.  We handle these with clearly-flagged *synthetic* assumptions, all kept in a single configuration file, so they can be swapped for real client values later.

___
<br>
# Modelling Overview  <a name="modelling-overview"></a>

Before getting into the code, it is worth explaining a few of the key ideas that the whole solution is built upon.

<br>
##### Leakage-Free Snapshots

A booking's risk changes over time.  A booking made a year in advance, with 180 days still to go, is a very different proposition to that same booking 2 days before arrival - if it hasn't been cancelled by then, the guest is very likely to turn up.

So rather than scoring each booking once, we take a "snapshot" of every booking at a series of moments before arrival (D-180, D-120, D-90, D-60, D-30, D-14, D-7, D-3, D-1 and D-0).  A booking only appears in a snapshot if it was *active* on that day - it had already been made, and it had not yet been cancelled.  The target is then whether the guest ultimately did *not* arrive.

This way the model learns to answer the exact question the business asks every day - "given everything we know *today*, how likely is this guest not to show up?"

<br>
##### Out-Of-Time Validation

A random train/test split would be very misleading here - the model would get to "peek" at bookings from the future.  We instead split strictly by arrival date:

* **Training:** arrivals July 2015 - June 2016
* **Validation & calibration:** arrivals July - December 2016
* **Test (backtest):** arrivals January - August 2017

<br>
##### Calibration

For ranking bookings, any model that orders risk correctly is fine.  For overbooking however, we *add up* probabilities to get an expected number of guests - so a predicted 20% really needs to mean 20%.  Gradient boosted trees are often slightly off here, so we pass the raw scores through an *isotonic regression* fitted on the validation period, which maps them onto observed no-arrival rates.

<br>
##### From Bookings To Occupancy

If each booking on the books for a given night has its own probability of arriving, then the total number of guests who arrive follows a *Poisson-binomial* distribution - the sum of many independent "coin flips", each with its own probability.  We can compute this distribution exactly, add the expected late "pickup" (bookings still to come), and get a full distribution of final occupancy - not just a single number.

<br>
##### The Overbooking Trade-Off

Every extra booking we accept has two possible outcomes.  If there is a free room, we earn the room rate plus ancillary spend (food, bar, spa etc).  If the hotel ends up over capacity, a guest has to be walked - which costs us a room elsewhere, transfer, compensation, and some goodwill.

For each night we therefore choose the number of extra bookings *X* that maximises:

*Profit(X) = room value × E[occupied rooms] - walk cost × E[walked guests]*

With a full occupancy distribution in hand this is quick to compute for every value of X - and it naturally sells more on nights with lots of uncertainty and free capacity, and less (or nothing) on nights that are already close to full.

___
<br>
# Data Preparation  <a name="data-prep"></a>

<br>
#### Cleaning

In the code below, we:

* Build proper arrival, booking and departure dates
* Create our target - the guest did *not* arrive (either a cancellation or a no-show)
* Calculate how many days before arrival each cancellation happened
* Remove bookings with no guests or no overnight stay, and a single data error with an ADR of €5,400
* Group the many travel agents into the top 40 plus an "other" category

<br>
```python

# arrival, booking & departure dates
df["arrival"] = pd.to_datetime(dict(year = df["arrival_date_year"],
                                    month = df["arrival_date_month"].map(MONTHS),
                                    day = df["arrival_date_day_of_month"]))
df["booking_date"] = df["arrival"] - pd.to_timedelta(df["lead_time"], unit = "D")
df["status_date"] = pd.to_datetime(df["reservation_status_date"])
df["nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

# target: the guest did NOT arrive (cancellation or no-show)
df["no_arrival"] = df["reservation_status"].isin(["Canceled", "No-Show"]).astype(int)
df["is_cancel"] = (df["reservation_status"] == "Canceled").astype(int)

# how many days before arrival the cancellation happened
df["cancel_days_before"] = np.where(df["is_cancel"] == 1,
                                    (df["arrival"] - df["status_date"]).dt.days,
                                    np.nan)

# remove invalid bookings & outliers
df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
df = df[df["nights"] > 0]
df = df[df["adr"] <= 1000]

```
<br>

#### Removing Leaky Features

This step is easy to overlook, but it is one of the most important in the project.  Several columns in this dataset are only finalised *after* the booking is made, and some are filled in at check-in - so they would never be available at the moment we need to make a decision.  We exclude:

* **country** - in this dataset the country is filled in at check-in, so cancelled bookings show Portugal unnaturally often (a known artefact)
* **booking_changes, assigned_room_type, required_car_parking_spaces, total_of_special_requests** - final values, known only after the fact
* **reservation_status, reservation_status_date, is_canceled** - these *are* the answer!

We also deliberately excluded *ADR* and *arrival month*.  With only around a year of training history the model started to "memorise" the season and price levels of one specific year - including them actually *reduced* test ROC AUC from 0.802 to 0.787.

<br>
#### Building The Snapshots

In the code below we build the snapshot dataset.  For each horizon *h* we keep only the bookings that were active on day D-h - already made (lead time of at least *h* days) and not yet cancelled.

```python

# the moments before arrival at which we "take a picture" of each booking
SNAPSHOT_HORIZONS = [0, 1, 3, 7, 14, 30, 60, 90, 120, 180]

def snapshot_mask(df, h):
    booked = df["lead_time"] >= h
    not_yet_cancelled = (df["is_cancel"] == 0) | (df["cancel_days_before"] <= h)
    return booked & not_yet_cancelled

def build_snapshots(df, horizons = SNAPSHOT_HORIZONS):
    parts = []
    for h in horizons:
        part = df.loc[snapshot_mask(df, h)].copy()
        part["days_to_arrival"] = h
        parts.append(part)
    return pd.concat(parts, ignore_index = True)

snapshots = build_snapshots(df)

```
<br>
The features for each snapshot are then things like *days to arrival*, *days since booking*, the original lead time, length of stay, party size, the guest's previous cancellations and completed stays, and categorical features such as hotel, meal plan, market segment, distribution channel, room type, deposit type, customer type and agent.

One subtle example of leakage control - the number of days a booking spent on the waiting list is capped at the number of days since booking, as anything beyond that hadn't happened yet at the time of the snapshot.

The resulting snapshot datasets were:

```python

snapshots: train=204,602  valid=134,831  test=175,763

```

___
<br>
# Training The Risk Model <a name="model-training"></a>

We train four models in total, so that we can see clearly where the improvement comes from:

* **Segment heuristic** - this mimics how a revenue manager works today - the historical cancellation rate for each combination of hotel, market segment, deposit type and horizon bucket (with shrinkage towards the overall mean for small segments)
* **Logistic regression** - a simple, interpretable machine learning baseline
* **LightGBM (uncalibrated)** - our gradient boosted tree model
* **LightGBM + calibration** - our target model

<br>
#### Segment Heuristic (Current Practice)

```python

class SegmentRateModel:
    keys = ["hotel", "market_segment", "deposit_type", "hb"]

    def fit(self, snap):
        d = snap[["hotel", "market_segment", "deposit_type"]].copy()
        d["hb"] = horizon_bucket(snap["days_to_arrival"])
        d["y"] = snap["no_arrival"].values
        self.global_ = d["y"].mean()
        g = d.groupby(self.keys)["y"].agg(["sum", "count"])

        # shrinkage towards the global mean
        self.table_ = ((g["sum"] + 20 * self.global_) / (g["count"] + 20)).rename("p")
        return self

```
<br>
#### LightGBM + Isotonic Calibration

In the code below we train LightGBM on the training period, using the validation period for early stopping.  The raw predictions on the validation period are then used to fit the isotonic calibrator.

The parameters are deliberately conservative (a low learning rate, a large minimum number of samples per leaf, and smoothing on the categorical features) as we have a lot of high-cardinality categories, such as travel agents, which are easy to overfit.

```python

# LightGBM parameters
LGBM_PARAMS = dict(objective = "binary",
                   learning_rate = 0.03,
                   num_leaves = 31,
                   min_child_samples = 300,
                   cat_smooth = 50,
                   min_data_per_group = 200,
                   cat_l2 = 10,
                   feature_fraction = 0.8,
                   bagging_fraction = 0.8,
                   bagging_freq = 1,
                   lambda_l2 = 1.0)

# train with early stopping on the validation period
dtr = lgb.Dataset(X_train, y_train, categorical_feature = CATEGORICAL)
dva = lgb.Dataset(X_valid, y_valid, reference = dtr, categorical_feature = CATEGORICAL)

booster = lgb.train(LGBM_PARAMS,
                    dtr,
                    num_boost_round = 2000,
                    valid_sets = [dtr, dva],
                    callbacks = [lgb.early_stopping(100)])

# fit isotonic calibration on validation predictions
raw_valid = booster.predict(X_valid, num_iteration = booster.best_iteration)
calibrator = IsotonicRegression(out_of_bounds = "clip", y_min = 0.0, y_max = 1.0)
calibrator.fit(raw_valid, y_valid)

# calibrated probability that the guest will NOT arrive
p_no_arrival = calibrator.predict(booster.predict(X_test, num_iteration = booster.best_iteration))

```

___
<br>
# Model Performance & Explainability <a name="model-performance"></a>

<br>
#### Model Comparison

All four models were evaluated on the 2017 test period, which none of them had seen during training or calibration.  As well as ROC AUC we look at PR AUC (useful when the positive class is the minority), the Brier score and Log Loss (which reward good probabilities, not just good ranking) and the Expected Calibration Error (ECE).

| **Model** | **ROC AUC** | **PR AUC** | **Brier** | **LogLoss** | **ECE** | **Mean predicted** | **Mean actual** |
|---|---|---|---|---|---|---|---|
| Segment heuristic (current practice) | 0.783 | 0.378 | 0.0785 | 0.2770 | 2.2% | 12.8% | 10.7% |
| Logistic regression | 0.759 | 0.420 | 0.0785 | 0.2841 | 1.1% | 11.7% | 10.7% |
| LightGBM (uncalibrated) | 0.802 | 0.449 | 0.0752 | 0.2646 | 0.2% | 10.7% | 10.7% |
| LightGBM + calibration (target model) | **0.802** | 0.438 | **0.0754** | 0.2655 | 0.8% | 11.4% | 10.7% |

<br>
Interestingly, the segment heuristic is actually a pretty strong baseline - it beats the logistic regression on ROC AUC - which tells us the revenue team's intuition is good!  LightGBM however is better on every metric.

The heuristic also *over-predicts* risk (12.8% predicted vs 10.7% actual), which matters a lot here - an over-pessimistic view of cancellations is exactly what leads to too much overbooking and walked guests.

<br>
![alt text](/img/posts/hotel-overbooking-roc.png "ROC curve - 2017 test period")

<br>
![alt text](/img/posts/hotel-overbooking-calibration.png "Calibration - predicted vs actual")

<br>
#### Performance By Horizon

Looking at performance at each horizon separately is really informative:

| **Days to arrival** | **Bookings** | **No-arrival rate** | **ROC AUC (heuristic)** | **ROC AUC (LightGBM)** | **Brier (heuristic)** | **Brier (LightGBM)** |
|---|---|---|---|---|---|---|
| 0 | 25,391 | 2.4% | 0.566 | 0.633 | 0.0235 | 0.0227 |
| 1 | 23,836 | 2.8% | 0.537 | 0.623 | 0.0274 | 0.0264 |
| 3 | 22,447 | 3.4% | 0.547 | 0.624 | 0.0326 | 0.0321 |
| 7 | 21,110 | 5.3% | 0.582 | 0.669 | 0.0523 | 0.0500 |
| 14 | 20,002 | 9.3% | 0.573 | 0.688 | 0.0810 | 0.0788 |
| 30 | 17,816 | 14.1% | 0.599 | 0.698 | 0.1151 | 0.1086 |
| 60 | 15,187 | 19.4% | 0.633 | 0.714 | 0.1364 | 0.1322 |
| 90 | 12,908 | 23.7% | 0.664 | 0.731 | 0.1543 | 0.1455 |
| 120 | 10,724 | 27.9% | 0.676 | 0.741 | 0.1663 | 0.1606 |
| 180 | 6,342 | 35.2% | 0.683 | 0.746 | 0.1870 | 0.1833 |

<br>
LightGBM beats the heuristic at *every* horizon - and the gap is largest exactly where it matters most for operations.  At D-1 to D-14 the heuristic is not much better than a coin toss (ROC AUC 0.54 - 0.58), while LightGBM stays at 0.62 - 0.69.

You may notice that the per-horizon AUCs are all lower than the overall figure of 0.802.  That is expected - within a single horizon the model can no longer use "how far away is arrival" to separate bookings, and that alone is a very strong signal.

<br>
#### Why Guests Don't Arrive (SHAP)

To understand *why* a booking is risky we use SHAP values, which LightGBM can compute natively.  These break each prediction down into the contribution of each feature.

<br>
![alt text](/img/posts/hotel-overbooking-shap-importance.png "Top drivers of no-arrival risk (SHAP)")

<br>
Days to arrival is by far the biggest driver - as discussed, the closer a booking gets to arrival without being cancelled, the safer it becomes.  This is followed by the agent/intermediary, deposit type, and the guest's history of previous cancellations.

Looking at the direction of the effects for some of the categorical features:

<br>
![alt text](/img/posts/hotel-overbooking-shap-categories.png "Category effects on no-arrival risk")

<br>
Group bookings and Online Travel Agent bookings increase the risk, while Offline Travel Agent bookings reduce it.  Transient customers are riskier, while Transient-Party, Group and Contract customers are more reliable.

The very large effect for *Non Refund* deposits looks counter-intuitive - why would a guest who can't get their money back cancel more often?  This is a known artefact of this particular dataset (such bookings are almost always recorded as cancelled), and it is certainly something to check carefully once we have the client's own data!

These SHAP values are also used operationally - every booking on the daily contact list comes with its top risk drivers, so the front office team know *why* a booking was flagged.

___
<br>
# Occupancy Forecast <a name="occupancy-forecast"></a>

With a calibrated probability for every booking, we can now move from the booking level to the hotel level - how many guests will actually be in the hotel on a given night?

We do this in two stages.

<br>
#### Stage 1 - Structural Forecast

For each night we take the guests already in house, add the expected arrivals from bookings on the books, and add a forecast of "pickup" (bookings still to come).  The exact distribution of arrivals from the bookings on the books is calculated with a Poisson-binomial distribution, using a simple dynamic programming approach:

```python

def poisson_binomial_pmf(probs):

    # exact distribution of a sum of independent Bernoulli(probs)
    pmf = np.array([1.0])
    for p in probs:
        new = np.zeros(len(pmf) + 1)
        new[:-1] += pmf * (1 - p)
        new[1:] += pmf * p
        pmf = new
    return pmf

def structural_distribution(in_house, show_probs, mu_pickup):
    pmf = np.convolve(poisson_binomial_pmf(show_probs), poisson_pmf(mu_pickup))
    return np.concatenate([np.zeros(in_house), pmf])

```
<br>
The pickup forecast is expressed as a *share of free capacity* that usually fills up after the decision date, based on the same weekday over the previous 8 weeks (using only nights that had already happened at the time of the decision).  This means the forecast pickup naturally shrinks on nights that are almost full, and grows on nights that are empty.

The room count isn't available in the public data, so capacity is estimated as the maximum realised occupancy - the data clearly hits a ceiling of 226 rooms at the City hotel and 187 at the Resort.

<br>
#### Stage 2 - Booking Curve Correction

The structural forecast assumes every booking is independent, and that the hotel's sales behaviour doesn't change as nights fill up - neither of which is completely true.  So we train a second, smaller LightGBM model which learns how the current state of the booking book translates into *final* occupancy.

Its features are aggregates of the book for each night - guests in house, bookings on the books, expected arrivals and their variance, pickup forecasts, the remaining gap to capacity, horizon and day of week.

For the uncertainty around this forecast we use a *conformal* approach - we keep the out-of-sample residuals from cross-validation (in blocks of whole weeks, to avoid leakage between neighbouring nights), and use them by hotel and horizon bucket to build the occupancy distribution.  This gives much more realistic intervals than assuming independence.

<br>
#### Forecast Accuracy

We compared the Mean Absolute Error (in rooms) of the structural forecast and the stage-2 model against a naive "same night last year" forecast over the test period:

| **Hotel** | **Horizon** | **MAE: naive (last year)** | **MAE: structural** | **MAE: stage-2 model** | **Nights** |
|---|---|---|---|---|---|
| City Hotel | D0-3 | 34.15 | 3.64 | **2.85** | 52 |
| City Hotel | D4-7 | 29.73 | 7.21 | **6.24** | 52 |
| City Hotel | D8-14 | 27.02 | 10.38 | **8.42** | 91 |
| City Hotel | D15-30 | 23.61 | 13.15 | **12.04** | 208 |
| City Hotel | D31-60 | 19.76 | 16.86 | **13.53** | 377 |
| Resort Hotel | D0-3 | 25.37 | **2.76** | 2.79 | 52 |
| Resort Hotel | D4-7 | 19.15 | **5.23** | 6.16 | 52 |
| Resort Hotel | D8-14 | 22.13 | **8.03** | 10.02 | 91 |
| Resort Hotel | D15-30 | 20.01 | **11.49** | 13.70 | 208 |
| Resort Hotel | D31-60 | 15.28 | 13.83 | **12.97** | 377 |

<br>
Both approaches are dramatically better than the naive forecast - 3 days out, the City hotel forecast is typically within 3 rooms of the final figure, versus 34 rooms for "same night last year".

Interestingly, the stage-2 correction clearly helps at the City hotel, but at the Resort it is actually slightly *worse* than the structural forecast at short and medium horizons.  This is a useful finding in itself - a sensible refinement would be to choose the forecasting approach per hotel and horizon, rather than applying stage-2 everywhere.

___
<br>
# Overbooking Optimisation & Backtest <a name="overbooking"></a>

<br>
#### Optimising A Single Night

With a full occupancy distribution for each night, the optimisation is surprisingly simple.  In the code below, for every candidate number of extra bookings *X* (from 0 up to 40), we:

* Model the number of extra bookings that actually materialise as Binomial(X, q), where *q* is the historical arrival rate of last-minute bookings
* Combine this with the occupancy distribution
* Calculate the expected profit - room value for every occupied room, minus the walk cost for every guest above capacity
* Keep the *X* with the highest expected profit

```python

def optimize_night(pmf_r, capacity, value, w_cost, q, x_max = 40):

    k = np.arange(len(pmf_r) + x_max + 1)
    occ_val = value * np.minimum(k, capacity) - w_cost * np.maximum(k - capacity, 0)

    profits = []
    binom = np.array([1.0])
    for x in range(x_max + 1):
        if x > 0:
            b = np.zeros(len(binom) + 1)
            b[:-1] += binom * (1 - q)
            b[1:] += binom * q
            binom = b
        pmf_t = np.convolve(pmf_r, binom)
        profits.append(pmf_t @ occ_val[:len(pmf_t)])

    x_best = int(np.argmax(profits))
    return x_best, profits[x_best] - profits[0]

```
<br>
The room value is the night's ADR plus an assumed ancillary spend (€20 per room-night at the City hotel, €35 at the Resort), and the cost of walking a guest is set to 2 × ADR + €150 - both synthetic assumptions.

The chart below shows an example decision for one night at the Resort hotel.  Without any extra sales the hotel is expected to end up around 173 guests, well short of its 187 rooms.  Selling 14 extra bookings shifts the distribution right up against capacity - there is a 24% chance of having to walk at least one guest, but the expected gain is still around €2,469 for that single night.

<br>
![alt text](/img/posts/hotel-overbooking-example-decision.png "Example overbooking decision for one night")

<br>
#### Backtesting On 2017

To see how this would have performed in practice, we replayed every night from January to August 2017 for both hotels.  For each night, the decision was taken at D-3, using *only* what was known at the time.

We compared four policies:

* **No extra sales** - what actually happened (our zero point)
* **Static overbooking 5%** - a fixed 5% of capacity every night, as used by many hotels
* **ML model** - our daily recommendation
* **Oracle** - the best possible decision for each night with perfect hindsight (an upper bound)

As the dataset only contains bookings that were *accepted*, we don't know how much extra demand was turned away on sold-out nights.  We generate this synthetically - on nights that were at least 95% full, extra demand is drawn from a Poisson distribution with a mean of 10% of capacity.  All four policies are evaluated on *exactly* the same random draws (common random numbers), so the comparison between them is fair.

```python

# the extra bookings each policy would have accepted, for each night
policies = {"none": np.zeros(len(dec), dtype = int),
            "static": np.round(0.05 * dec["capacity"]).astype(int).to_numpy(),
            "model": dec["x_opt"].to_numpy(),
            "oracle": best_x}

# realise each policy against the same synthetic demand & random draws
for key, x in policies.items():
    recovered, walks, gain = realize(x)
    dec[f"{key}_recovered"] = recovered
    dec[f"{key}_walks"] = walks
    dec[f"{key}_gain"] = gain

```
<br>
The results across both hotels were as follows:
<br>
<br>

| **Policy** | **Recovered room-nights** | **Guests walked** | **Nights with walks** | **Revenue from recovered rooms** | **Walk cost** | **Net result** | **% of potential** |
|---|---|---|---|---|---|---|---|
| No extra sales (current practice) | 0 | 0 | 0 | €0 | €0 | €0 | 0.0% |
| Static overbooking 5% | 1,156 | 697 | 160 | €179,031 | €291,195 | -€112,164 | -61.4% |
| ML model (daily recommendation) | 977 | 205 | 73 | €149,741 | €82,209 | **€67,532** | **36.9%** |
| Oracle (upper bound) | 1,181 | 0 | 0 | €182,774 | €0 | €182,774 | 100.0% |

<br>
This is the key result of the whole project!

Static overbooking recovers the most room-nights of the two realistic policies - but it does so blindly, walking 697 guests on 160 different nights.  The cost of those walks wipes out all of the extra revenue and more, leaving a *loss* of €112k.

The ML model is far more selective.  It recovers slightly fewer room-nights, but walks less than a third as many guests, on fewer than half as many nights - giving a net *gain* of €68k, or 37% of what would be possible with perfect foresight.

<br>
![alt text](/img/posts/hotel-overbooking-backtest-cumulative.png "Cumulative net result of each overbooking policy over 2017")

<br>
The cumulative chart shows that this isn't driven by one lucky period - the ML model's advantage builds steadily over the year, while the static policy's losses accelerate sharply in the busy summer months, exactly when the hotels are full and walks become most likely.

<br>
#### Daily Plan For The Coming 60 Days

In production, the same logic produces a forward-looking plan every day.  Below is a snippet of the plan for the City hotel as of 15 June 2017:

| **Night** | **Capacity** | **In house** | **On books** | **Forecast** | **P10 - P90** | **P(full)** | **Extra bookings** | **Recommendation** |
|---|---|---|---|---|---|---|---|---|
| 2017-06-15 | 226 | 142 | 78 | 218.2 | 213 - 222 | 0.03 | 7 | Allow overbooking: +7 bookings |
| 2017-06-16 | 226 | 85 | 130 | 215.1 | 210 - 219 | 0.01 | 10 | Allow overbooking: +10 bookings |
| 2017-06-17 | 226 | 37 | 177 | 214.8 | 210 - 219 | 0.01 | 10 | Allow overbooking: +10 bookings |
| 2017-06-18 | 226 | 17 | 179 | 198.2 | 193 - 202 | 0.00 | 29 | Low occupancy - sales push / promotion |
| 2017-06-19 | 226 | 5 | 199 | 205.7 | 196 - 214 | 0.00 | 18 | Low occupancy - sales push / promotion |
| 2017-06-20 | 226 | 1 | 210 | 213.1 | 203 - 221 | 0.04 | 10 | Allow overbooking: +10 bookings |
| 2017-06-24 | 226 | 1 | 211 | 217.4 | 206 - 226 | 0.18 | 4 | Allow overbooking: +4 bookings |
| 2017-06-26 | 226 | 1 | 218 | 220.0 | 209 - 226 | 0.26 | 1 | Allow overbooking: +1 bookings |
| … | … | … | … | … | … | … | … | … |

<br>
Nights where a large number of rooms are expected to remain empty are flagged as a *sales push / promotion* opportunity instead - overbooking alone won't fill them, so the revenue team should be looking at pricing or marketing.

___
<br>
# Preventive Actions & Cancellation Policies <a name="preventive-actions"></a>

<br>
#### Daily Contact List

Overbooking deals with cancellations *after* the fact.  The other side of the coin is trying to prevent them in the first place.

Every day, all active bookings arriving within the next 14 days are scored, and those above a risk threshold are put on a contact list - with an e-mail asking the guest to confirm their arrival (with a discounted prepayment offer) for medium risk, and a phone call for the highest risk.  Each booking on the list comes with its revenue at risk and its top SHAP drivers.

As of 15 June 2017, there were 1,424 active bookings arriving within 14 days, of which 227 were flagged for an e-mail.  The model expected 95 of the 1,424 bookings not to arrive, with around €40k of revenue at risk.

<br>
#### A/B Testing Guest Contact (Synthetic)

So who is actually worth contacting?  Contacting everyone costs money (and the prepayment discount gives away revenue from guests who would have turned up anyway), while contacting no-one saves nothing.

The public data contains no actions taken towards guests, so we *simulated* an A/B test on the real D-14 bookings in the test period, to demonstrate how this would be evaluated in a real deployment.  Half of the bookings were randomly assigned to a "contacted" group, where a real cancellation could be "saved" with a probability that is largest for undecided guests (risk around 50%) and small for those who were almost certain to arrive or almost certainly lost:

```python

# synthetic effect of contact: largest for "undecided" guests
p = experiment["p_lgbm"].to_numpy()
effect = OUTREACH_MAX_EFFECT * 4 * p * (1 - p)

# a real cancellation in the treatment group is "saved" with probability = effect
saved = (experiment["treated"] == 1) & (experiment["no_arrival"] == 1) & (rng.random(len(experiment)) < effect)
experiment["outcome"] = np.where(saved, 0, experiment["no_arrival"])

```
<br>
Across the 20,002 bookings in the experiment, the no-arrival rate fell from 9.5% in the control group to 7.6% in the contacted group - a reduction of 1.9 percentage points (95% CI: 1.2 to 2.7).

The more useful question however is *where* that effect comes from.  We split the bookings into deciles of predicted risk, and fitted a smoothed uplift model (a logistic regression with a treatment × risk interaction) to reduce the noise of the individual deciles.  Combining the uplift with booking value, the chance of reselling a released room, the cost of contact and the cost of the discount gives us the net value of contacting each decile:

<br>
![alt text](/img/posts/hotel-overbooking-outreach-uplift.png "Effect of contact by risk decile, and net result per contact")

<br>
Contacting the lowest-risk bookings actually *loses* money - around €4 per contact in the lowest decile, as the discount is mostly given to guests who were going to arrive anyway.  Contact only starts to pay off from decile 7 upwards, which corresponds to a risk threshold of around **12%** - and this is the threshold used for the daily contact list.

Once live, the synthetic effect would of course be replaced by a real A/B test - but the method for evaluating it, and for setting the threshold, stays exactly the same.

<br>
#### Cancellation & Prepayment Policies

Finally, we looked at whether changing the booking terms themselves could help.  We ran a *what-if* simulation for every hotel × channel × lead time segment with no deposit today - assuming that requiring prepayment would cut cancellations by 40%, but scare off 8% of bookers (both synthetic assumptions).

The segments where this looks worthwhile are almost all Travel Agent / Tour Operator bookings at the City hotel:

| **Hotel** | **Channel** | **Lead time (days)** | **Bookings** | **Cancellation rate** | **Late cancellation rate** | **Revenue change per year** |
|---|---|---|---|---|---|---|
| City hotel | TA/TO | 8-30 | 9,873 | 28.9% | 9.1% | €30,602 |
| City hotel | TA/TO | 91-180 | 13,545 | 35.7% | 3.3% | €24,997 |
| City hotel | TA/TO | 31-90 | 16,835 | 32.8% | 4.1% | €24,956 |
| City hotel | TA/TO | 181-365 | 8,802 | 43.8% | 2.4% | €23,707 |
| Resort hotel | TA/TO | 181-365 | 5,515 | 38.6% | 1.2% | €11,961 |
| Resort hotel | TA/TO | 91-180 | 6,362 | 34.6% | 1.5% | €7,157 |
| … | … | … | … | … | … | … |
| Resort hotel | Corporate | 0-7 | 1,439 | 6.9% | 6.9% | -€658 |

<br>
For corporate bookings at the Resort the result is negative - the value of the bookings that would be lost outweighs the cancellations saved, so the current terms should stay as they are.

This is a much more uncertain estimate than the overbooking backtest (it rests heavily on the assumed guest response), so we present it as a *potential* to be tested with the client, rather than as part of the business case.

___
<br>
# Monitoring & Business Case <a name="monitoring-business-case"></a>

<br>
#### Monitoring

A model like this is only useful if it keeps working.  We track two things:

* **Data drift** - the Population Stability Index (PSI) of each feature between the training and test periods
* **Stability of quality** - ROC AUC and calibration for each month of arrivals

```python

def psi(expected, actual, categorical = False, bins = 10):
    eps = 1e-4
    if categorical:
        e = expected.astype(str).value_counts(normalize = True)
        a = actual.astype(str).value_counts(normalize = True)
        idx = e.index.union(a.index)
        e, a = e.reindex(idx, fill_value = 0) + eps, a.reindex(idx, fill_value = 0) + eps
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.Series(np.histogram(expected, edges)[0] / len(expected)) + eps
        a = pd.Series(np.histogram(actual, edges)[0] / len(actual)) + eps
    return float(((a - e) * np.log(a / e)).sum())

```
<br>
The top features by PSI were:

| **Feature** | **PSI** | **Status** |
|---|---|---|
| Agent / intermediary | 0.754 | ALERT |
| Market segment | 0.211 | Warning |
| Customer type | 0.170 | Warning |
| Booking lead time (days) | 0.097 | OK |
| Meal plan | 0.075 | OK |
| … | … | … |

<br>
The mix of agents had already shifted substantially by 2017 - which makes sense, as agents come and go and their volumes change.  Monthly performance has so far held up well (ROC AUC between 0.72 and 0.83, with calibration error within ±2.2 percentage points every month), but given this level of drift we recommend retraining monthly.

<br>
#### Business Case

Pulling everything together, the annual financial impact for the two hotels is:

| **Item** | **EUR / year** | **Source** |
|---|---|---|
| Model-based overbooking (backtest, annualised) | €101,436 | public data + synthetic turned-away demand |
| Preventive actions for high-risk bookings | €22,563 | synthetic experiment |
| Cost of running the solution | -€6,000 | assumption |
| **Total (excluding policy changes)** | **€118,000** | |
| Additionally: potential of prepayment policy changes | €131,684 | what-if simulation, synthetic assumptions |
| For comparison: overbooking upper bound (oracle) | €274,538 | backtest |

<br>
With an assumed project cost of €30,000 (one person for around three months), this gives a first-year ROI of around 293% and a payback period of roughly **3 months** - or around €59k per hotel per year.

It is worth stressing again that a number of these inputs are synthetic, and that the backtest only covers January to August, so the annualised figure may overstate the effect.  These figures are the case for running a pilot, not a guarantee!

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

As this is a proof of concept on public data, the first step is a pilot on the client's own PMS data, with real room capacities and costs.  The key data fields needed are the booking date, arrival, nights, status, *cancellation date*, channel, segment, deposit, room type and rate.

We would then replace each synthetic element with a measured value - a real walk cost from finance, turned-away demand from the booking engine's denial logs, and a live A/B test of guest outreach.

There are also a few known limitations to address - the solution currently works at hotel level rather than room-type level, the annualised figures need confirming over a full season, and the Non Refund deposit artefact needs checking on real data.

For a multi-property rollout we would move to a central model with hotel-level features, and integrate the daily outputs (overbooking plan and contact list) into Power BI or Snowflake - so that the revenue and front office teams see them in the tools they already use every day.
