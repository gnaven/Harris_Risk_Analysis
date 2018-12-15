"""
This script imports all three datasets and cleans them,
bins them data according to the user-specified binning scheme, 
aggregates them on SupplierID and category, joins them and 
exports them to the prepared_data directory.

Given:
* binning scheme (five, three, or bin)

Creates:
prepared_data/join_*_scale.csv

Example:
python data_prep.py five
"""
import pandas as pd
import numpy as np
import sys

# ********************** get parameters *********************
if len(sys.argv) != 2:
	print "Please specify the binning scale ('five', 'three', 'bin')"
	exit()
scale = sys.argv[1]
if scale != "five" and scale != "three" and scale != "bin":
	print scale + "is not a valid binning scale"
	exit()

print "Binning scale: " + scale



# ********************** import data *********************
print "import data"

purchasing = pd.read_csv('original_data/Purchasing_Tool_FY16-FY18_2018-11-16.csv')
expo = pd.read_csv('original_data/Expo_Archive_2018-11-16.csv')
scorecard = pd.read_excel('original_data/Expo_Supplier_Scorecard_Data(1).xlsx')

# save dataframe with part number and category
category = pd.DataFrame()
category['Part Number'] = purchasing['Part']
category['Category'] = purchasing['Category']
category = category.drop_duplicates(subset = ['Part Number'])
category = category.dropna()

# save dataframe with 'Supplier ID' and 'SupplierID'
supplier = pd.DataFrame()
supplier['Supplier ID'] = purchasing['Supplier ID']
supplier['SupplierID'] = purchasing['SupplierID']



# ********************* clean data *********************
print "clean"

# drop irrelevant features
purchasing = purchasing.drop(['SupplierID', 'Shipment', 'PO', 'Line',
		'Release','BuyerNumber',
		'BuyProgram', 'BuyerName', 'Description', 'ReceiverNumber',
		'Year', 'Period', 'Week', 'Qtr', 'CommodityLevel1',
		'CommodityLevel2', 'CommodityLevel3',
		'ForecastType', 'PaymentTerm', 'SupplierClassification',
		'PoPlacedDate', 'PoPlacedYear',
		'Org_Code', 'SupplierSite'], axis = 1)

expo = expo.drop(['Revision', 'Part Description', 'PO Number',
		'Line', 'Schedule', 'Revision',
		'Part Revision', 'Supplier Due Date',
		'Message', 'Qty Open', 'Unit Price',
		'Extended Price', 'VMI', 'Date Modified', 'Comments',
		'Harris Comments', 'Shipments', 'Business Unit Code'],
		axis =1)

# remove irrelevant columns from scorecard
del scorecard['QualityRating']
del scorecard['DeliveryRating']
del scorecard['TotalRating']
del scorecard['TotalScore']



# ********************* remove na *********************
# drop na in expo
expo = expo.dropna()
expo = expo.reset_index(drop = True)

# drop na from purchasing tool
purchasing = purchasing.dropna()
purchasing = purchasing.reset_index(drop = True)

# drop na from scorecard
scorecard = scorecard.dropna()
scorecard = scorecard.reset_index(drop = True)


# ********************* data manipulation *********************
# fix date format in expo
from datetime import datetime as dt
def convert_to_date(s):
    try:
        return dt.date(dt.strptime(str(s)[:10], '%Y-%M-%d'))
    except (TypeError, ValueError) as e:
        return (pd.NaT)

# confirmed dock date has a slightly different format than
# the other dates
def convert_to_date2(s):
    try:
        return dt.date(dt.strptime(str(s), '%m/%d/%Y'))
    except (TypeError, ValueError) as e:
        return (pd.NaT)

expo['Confirmed Dock Date'] = expo['Confirmed Dock Date'].apply(convert_to_date2)
expo['Performance Date'] = expo['Performance Date'].apply(convert_to_date)
expo['Need Date'] = expo['Need Date'].apply(convert_to_date)
expo['Order Date'] = expo['Order Date'].apply(convert_to_date)

# transform ABC column in purchasing
ABC_dict = {'A':0.7, 'B':0.5, 'C':0.3}
purchasing['ABC'] = [ABC_dict[d] for d in purchasing['ABC']]

# fix purchase lead time codes
# convert codes to number of days
purch_lead_time_dict = {240:183, 299:365 ,999:365, 365:274, 364:274, 350:274}
purchasing['PurchLeadTime'] = [purch_lead_time_dict[i] if i in
	purch_lead_time_dict.keys() else i for i in purchasing['PurchLeadTime']]

# keep only month ratings in scorecard
# periods are not fixed length; some are a year and some
# are a month. The years often overlap with the months.
periods = ['Previous Fiscal Year', 'Current Fiscal Year', 
		'Current Period', 'Last 12 Complete Periods']
index = []
for i in range(len(scorecard)):
    if scorecard.iloc[i]['Period'] in periods:
        index += [i]
scorecard = scorecard.drop(index, axis=0)
del scorecard['Period']



# ********************* feature engineering *********************
# engineer late and early columns in expo
expo['temp_Late'] =  expo['Performance Date'] - expo['Confirmed Dock Date']
expo['temp_Late'] = expo['temp_Late'].astype('timedelta64[D]')

expo['Late'] = [i if i > 0 else 0 for i in expo['temp_Late']]
expo['Early'] = [abs(i) if i < 0 else 0 for i in expo['temp_Late']]

del expo['temp_Late']


# create return spend and return quantity columns. Zero out negatives in spend and qty
# return is binary feature: 0 = order, 1 = return
purchasing['return'] = [1 if d < 0 else 0 for d in purchasing['Spend']]

purchasing['Return_spend'] = [0 if d >= 0 else abs(d) for d in purchasing['Spend']]
purchasing['Spend'] = [d if d >= 0 else 0 for d in purchasing['Spend']]

purchasing['Return_qty'] = [0 if d >= 0 else abs(d) for d in purchasing['Qty']]
purchasing['Qty'] = [d if d >= 0 else 0 for d in purchasing['Qty']]

# add single/multi source column to p_tool as 'Source'.
source = pd.DataFrame()
source['Source'] = purchasing.groupby(['Part'])['Supplier ID'].apply(lambda x: x.drop_duplicates().count())
source['Part'] = source.index
source.index = range(len(source))
purchasing = purchasing.merge(source, on = ['Part'], how = 'left')


# generate unexpected cost = PoCost/StdCost
# measures percent change in expected cost and actual cost
purchasing['Unexp_Cost'] = np.nan
pd.to_numeric(purchasing['Unexp_Cost'], downcast='float')

for i, row in purchasing.iterrows():
    if (purchasing['StdCost'][i] > 0):
        k = (purchasing['PoCost'][i] / purchasing['StdCost'][i])
    else:
        k = 1
    purchasing.set_value(i,'Unexp_Cost',k)



# ********************* drop outliers *********************
# drop early/late outliers
# values above 5000 days seem to be typos; for example
# some of them had a year of '2106' instead of '2016'
expo = expo.drop(list(expo[expo['Late'] > 5000].index), axis = 0)
expo = expo.drop(list(expo[expo['Early'] > 5000].index), axis = 0)
expo = expo.reset_index(drop = True)



# ********************* binning *********************
print "binning"

if scale == "bin":
	# function to bin purchasing_tool on a 1-2 scale
	def bin_purchasing(d):
		data = d.copy()
		data['Unexp_Cost'] = pd.to_numeric(pd.cut(data['Unexp_Cost'],
				bins = [-1,1.0001,10000000], labels = [1,2]))
		data['StdCost'] = pd.to_numeric(pd.cut(data['StdCost'],
				bins = [-1,5,10000000], labels = [1,2]))
		data['PoCost'] = pd.to_numeric(pd.cut(data['PoCost'],
				bins = [-1,5,10000000], labels = [1,2]))
		data['Qty'] = pd.to_numeric(pd.cut(data['Qty'],
				bins = [-1,150,10000000], labels = [1,2]))
		data['InternalCostSavings'] = pd.to_numeric(pd.cut(data['InternalCostSavings'], 
				bins = [-1000000,0,10000000], labels = [2,1]))
		data['Spend'] = pd.to_numeric(pd.cut(data['Spend'],
				bins = [-1,5000,100000000], labels = [1,2]))
		data['PurchLeadTime'] = pd.to_numeric(pd.cut(data['PurchLeadTime'], 
				bins = [-1,100,10000], labels = [1,2]))
		data['ABC'] = pd.to_numeric(pd.cut(data['ABC'],
				bins = [0,0.6,1], labels = [1,2]))
		data['Return_spend'] = pd.to_numeric(pd.cut(data['Return_spend'], 
				bins = [-1,20,100000000], labels = [1,2]))
		data['Source'] = pd.to_numeric(pd.cut(data['Source'],
				bins = [-1,1.1,10], labels = [2,1]))
		return data

	# function to bin expo_archive on a 1-2 scale
	def bin_expo(d):
		data = d.copy()
		data['Late_bin'] = pd.to_numeric(pd.cut(data['Late'],
				bins = [-1,0.5,5000], labels = [1,2]))
		data['Early_bin'] = pd.to_numeric(pd.cut(data['Early'],
				bins = [-1,0.5,5000], labels = [1,2]))
		return data
elif scale == "three":
	def bin_purchasing(d):
		data = d.copy()
		data['Unexp_Cost'] = pd.to_numeric(pd.cut(data['Unexp_Cost'], 
			bins = [-1,.9999,2,10000000], labels = [1,2,3]))
		data['StdCost'] = pd.to_numeric(pd.cut(data['StdCost'], 
			bins = [-1,1,100,10000000], labels = [1,2,3]))
		data['PoCost'] = pd.to_numeric(pd.cut(data['PoCost'], 
			bins = [-1,1,100,10000000], labels = [1,2,3]))
		data['Qty'] = pd.to_numeric(pd.cut(data['Qty'], 
			bins = [-1,50,5000,10000000], labels = [1,2,3]))
		data['InternalCostSavings'] = pd.to_numeric(pd.cut(data['InternalCostSavings'], 
			bins = [-1000000,-100,10,10000000], labels = [3,2,1]))
		data['Spend'] = pd.to_numeric(pd.cut(data['Spend'], 
			bins = [-1,2500,100000,100000000], labels = [1,2,3]))
		data['PurchLeadTime'] = pd.to_numeric(pd.cut(data['PurchLeadTime'], 
			bins = [-1,50,350,10000], labels = [1,2,3]))
		data['ABC'] = pd.to_numeric(pd.cut(data['ABC'], 
			bins = [0,0.4,0.6,1], labels = [1,2,3]))
		data['Return_spend'] = pd.to_numeric(pd.cut(data['Return_spend'], 
			bins = [-1,20,500,100000000], labels = [1,2,3]))
		data['Source'] = pd.to_numeric(pd.cut(data['Source'], 
			bins = [-1,1.1,10], labels = [3,1]))

		return data

	def bin_expo(d):
		data = d.copy()
		data['Late_bin'] = pd.to_numeric(pd.cut(data['Late'], 
			bins = [-1,0.5,20,1000], labels = [1,2,3]))
		data['Early_bin'] = pd.to_numeric(pd.cut(data['Early'], 
			bins = [-1,0.5,20,1000], labels = [1,2,3]))
		return data
elif scale == "five":
	def bin_purchasing(d):
		data = d.copy()
		data['Unexp_Cost'] = pd.to_numeric(pd.cut(data['Unexp_Cost'],
				bins = [-1,.9999,1.0001,1.2,2,10000000], labels = [1,2,3,4,5]))
		data['StdCost'] = pd.to_numeric(pd.cut(data['StdCost'], 
				bins = [-1,1,5,50,100,10000000], labels = [1,2,3,4,5]))
		data['PoCost'] = pd.to_numeric(pd.cut(data['PoCost'],
				bins = [-1,1,5,50,100,10000000], labels = [1,2,3,4,5]))
		data['Qty'] = pd.to_numeric(pd.cut(data['Qty'],
				bins = [-1,50,150,500,5000,10000000], labels = [1,2,3,4,5]))
		data['InternalCostSavings'] = pd.to_numeric(pd.cut(data['InternalCostSavings'], 
				bins = [-1000000,-100,-10,0,10,10000000], labels = [5,4,3,2,1]))
		data['Spend'] = pd.to_numeric(pd.cut(data['Spend'], 
				bins = [-1,2500,10000,50000,100000,100000000], labels = [1,2,3,4,5]))
		data['PurchLeadTime'] = pd.to_numeric(pd.cut(data['PurchLeadTime'], 
				bins = [-1,50,125,225,350,10000], labels = [1,2,3,4,5]))
		data['ABC'] = pd.to_numeric(pd.cut(data['ABC'], 
				bins = [0,0.4,0.6,1], labels = [1,3,5]))
		data['Return_spend'] = pd.to_numeric(pd.cut(data['Return_spend'], 
				bins = [-1,20,500,1000,5000,100000000], labels = [1,2,3,4,5]))
		data['Source'] = pd.to_numeric(pd.cut(data['Source'],
				bins = [-1,1.1,3.1,10], labels = [5,2,1]))

		return data

	def bin_expo(d):
		data = d.copy()
		data['Late_bin'] = pd.to_numeric(pd.cut(data['Late'],
			bins = [-1,0.5,3,10,20,1000], labels = [1,2,3,4,5]))
		data['Early_bin'] = pd.to_numeric(pd.cut(data['Early'], 
			bins = [-1,0.5,3,10,20,1000], labels = [1,2,3,4,5]))
		return data



purch_binned = bin_purchasing(purchasing)
expo_binned = bin_expo(expo)

# ********************* aggregation *********************
print "aggregate"

# function to aggregate purchasing tool
# takes the dataframe and the columns to group by
def purchasing_aggregate(data, groupby):
    aggregate = pd.DataFrame()

    aggregate['mean_unexpcost'] = data.groupby(groupby)['Unexp_Cost'].mean()
    aggregate['median_unexpcost'] = data.groupby(groupby)['Unexp_Cost'].median()

    aggregate['mean_stdcost'] = data.groupby(groupby)['StdCost'].mean()
    aggregate['median_stdcost'] = data.groupby(groupby)['StdCost'].median()

    aggregate['mean_pocost'] = data.groupby(groupby)['PoCost'].mean()
    aggregate['median_pocost'] = data.groupby(groupby)['PoCost'].median()

    aggregate['mean_qty'] = data.groupby(groupby)['Qty'].mean()
    aggregate['median_qty'] = data.groupby(groupby)['Qty'].median()

    aggregate['mean_costsave'] = data.groupby(groupby)['InternalCostSavings'].mean()
    aggregate['median_costsave'] = data.groupby(groupby)['InternalCostSavings'].median()

    aggregate['mean_spend'] = data.groupby(groupby)['Spend'].mean()
    aggregate['median_spend'] = data.groupby(groupby)['Spend'].median()

    aggregate['mean_return_spend'] = data.groupby(groupby)['Return_spend'].mean()
    aggregate['median_return_spend'] = data.groupby(groupby)['Return_spend'].median()

    aggregate['mean_leadtime'] = data.groupby(groupby)['PurchLeadTime'].mean()
    aggregate['median_leadtime'] = data.groupby(groupby)['PurchLeadTime'].median()

    aggregate['mean_abc'] = data.groupby(groupby)['ABC'].mean()
    aggregate['median_abc'] = data.groupby(groupby)['ABC'].median()


    aggregate['mean_source'] = data.groupby(groupby)['Source'].mean()
    aggregate['median_source'] = data.groupby(groupby)['Source'].median()


    aggregate['Supplier ID'] = aggregate.index
    aggregate.index = range(len(aggregate))

    if 'Category' in groupby:
        aggregate['Category'] = [d[1] for d in aggregate['Supplier ID']]
        aggregate['Supplier ID'] = [d[0] for d in aggregate['Supplier ID']]

    return aggregate


# function to aggregate expo_archive
# takes a dataframe and the columns to group by
def expo_aggregate(data, groupby):
    new_expo = pd.DataFrame()
    new_expo['mean_late'] = data.groupby(groupby)['Late_bin'].mean()
    new_expo['median_late'] = data.groupby(groupby)['Late_bin'].median()

    new_expo['mean_early'] = data.groupby(groupby)['Early_bin'].mean()
    new_expo['median_early'] = data.groupby(groupby)['Early_bin'].median()

    new_expo['num_critical'] = data.groupby(groupby)['Critical'].apply(lambda x: x[x == True].count())
    new_expo['num_transactions_expo'] = data.groupby(groupby)['Late'].count()

    new_expo['%_critical'] = new_expo['num_critical'] / new_expo['num_transactions_expo']

    new_expo = new_expo.drop(['num_critical'], axis=1)
    new_expo = new_expo.drop(['num_transactions_expo'], axis=1)

    new_expo['Supplier ID'] = new_expo.index
    new_expo.index = range(len(new_expo))

    if 'Category' in groupby:
        new_expo['Category'] = [d[1] for d in new_expo['Supplier ID']]
        new_expo['Supplier ID'] = [d[0] for d in new_expo['Supplier ID']]

    return new_expo


# append 'Category' to expo
expo_binned_aug = expo_binned.merge(category, on = ['Part Number'], how = 'left')
expo_binned_aug = expo_binned_aug.dropna()

# aggregate purchasing_tool and expo_archive
agg_purch = purchasing_aggregate(purch_binned, ['Supplier ID', 'Category'])
agg_expo = expo_aggregate(expo_binned_aug, ['Supplier ID', 'Category'])


# second round of binning for expo_archive
# bins the aggregated values for critical
if scale == "bin":
	def bin_expo2(d):
		data = d.copy()
		data['%Critical'] = pd.to_numeric(pd.cut(data['%_critical'], 
			bins = [-1,0.2,1.1], labels = [1,2]))
		return data
elif scale == "three":
	def bin_expo2(d):
		data = d.copy()
		data['%Critical'] = pd.to_numeric(pd.cut(data['%_critical'], 
			bins = [-1,0.2,0.8,1.1], labels = [1,2,3]))
		return data
elif scale == "five":
	def bin_expo2(d):
		data = d.copy()
		data['%Critical'] = pd.to_numeric(pd.cut(data['%_critical'], 
			bins = [-1,0.2,0.4,0.6,0.8,1.1], labels = [1,2,3,4,5]))
		return data


agg_expo_bin = bin_expo2(agg_expo)
del agg_expo_bin['%_critical']


# *************** binning and aggregation for scorecard ***************
# bin scorecard on a 1-2 scale and aggregate
if scale == "bin":
	rating_dict = {5:1, 4:1, 3:2, 2:2, 1:2}
elif scale == "three":
	rating_dict = {5:1, 4:2, 3:2, 2:2, 1:3}

elif scale == "five":
	rating_dict = {5:1, 4:2, 3:3, 2:4, 1:5}


scorecard['Quality'] = [rating_dict[i] for i in scorecard['QualityScore']]
scorecard['Delivery'] = [rating_dict[i] for i in scorecard['DeliveryScore']]

del scorecard['QualityScore']
del scorecard['DeliveryScore']

scorecard_agg = pd.DataFrame();

scorecard_agg['quality_risk_mean'] = scorecard.groupby(['Vendor Code'])['Quality'].mean()
scorecard_agg['quality_risk_median'] = scorecard.groupby(['Vendor Code'])['Quality'].median()
scorecard_agg['delivery_risk_mean'] = scorecard.groupby(['Vendor Code'])['Delivery'].mean()
scorecard_agg['delivery_risk_median'] = scorecard.groupby(['Vendor Code'])['Delivery'].median()

scorecard_agg['SupplierID'] = scorecard_agg.index
scorecard_agg.index = range(len(scorecard_agg))



# ********************* join *********************
print "join"

join = agg_purch.merge(agg_expo_bin, on = ['Supplier ID', 'Category'], how = 'inner')

# append 'SupplierID' to join
# this is needed to join the joined table to scorecard
supplier = supplier.drop_duplicates(subset = ['Supplier ID'])
join = join.merge(supplier, on = ['Supplier ID'], how = 'inner')
join = join.sort_values(by = ['SupplierID'])

# join the joined table with scorecard
join_with_scorecard = join.merge(scorecard_agg, on = ["SupplierID"], how = 'inner')
del join_with_scorecard['Supplier ID']



# ********************* export *********************
print "export"

if scale == "bin":
	join_with_scorecard.to_csv("prepared_data/join_binary_scale.csv", index = False)
elif scale == "three":
	join_with_scorecard.to_csv("prepared_data/join_three_scale.csv", index = False)
elif scale == "five":
	join_with_scorecard.to_csv("prepared_data/join_five_scale.csv", index = False)


# ********************* finish *********************
print "completed"
