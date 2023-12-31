import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

df = pd.read_csv("Groceries data.csv")
df['Date'] = pd.to_datetime(df['Date'])

df["month"] = df['Date'].dt.month
df["day"] = df['Date'].dt.weekday

df["month"].replace([i for i in range(1, 12 + 1)], ["January", "February", "March", "April",
                    "May", "June", "July", "August", "September", "October", "November", "December"], inplace=True)
df["day"].replace([i for i in range(6 + 1)], ["Monday", "Tuesday",
                  "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], inplace=True)

# Filter the data based on User Inputs


def get_data(month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No result!"


def user_input_features():
    item = st.sidebar.selectbox("Item", ['tropical fruit', 'whole milk', 'pip fruit', 'other vegetables',
                                         'rolls/buns', 'citrus fruit', 'beef', 'frankfurter',
                                         'chicken', 'butter', 'fruit/vegetable juice',
                                         'packaged fruit/vegetables', 'chocolate', 'specialty bar',
                                         'butter milk', 'bottled water', 'yogurt', 'sausage', 'brown bread',
                                         'hamburger meat', 'root vegetables', 'pork', 'pastry',
                                         'canned beer', 'berries', 'coffee', 'misc. beverages', 'ham',
                                         'turkey', 'curd cheese', 'red/blush wine',
                                         'frozen potato products', 'flour', 'sugar', 'frozen meals',
                                         'herbs', 'soda', 'detergent', 'grapes', 'processed cheese', 'fish',
                                         'sparkling wine', 'newspapers', 'curd', 'pasta', 'popcorn',
                                         'finished products', 'beverages', 'bottled beer', 'dessert',
                                         'dog food', 'specialty chocolate', 'condensed milk', 'cleaner',
                                         'white wine'])
    month = st.sidebar.select_slider("Month", [
                                     "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day = st.sidebar.select_slider(
        'Day', ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], value="Sat")

    return month, day, item


month, day, item = user_input_features()

data = get_data(month, day)

st.text("")
st.text("")
try:
    st.title("Association Market Analyst MBA")
    st.title("M Daffa Alfikri")
    st.title("NIM : 211351076")
    st.text("Dataset:")
    st.dataframe(data)
except:

    st.markdown("""
    <div id="ifno-result">
      <p>Here are some input values to give a try!</p>
      <ul style="margin: 0 auto">
        <li>Month: &nbsp;<i>Jan</i></li>
        <li>Day: &nbsp;<i>Sun</i></li>
      </ul>
    </div>
  """, unsafe_allow_html=True)


# ==========================================================================================================================================================================

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1


if type(data) != type("No result!"):
    item_count = data.groupby(['Member_number', 'itemDescription'])[
        "itemDescription"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(
        index='Member_number', columns='itemDescription', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.01  # atau 1%
    frequent_items = apriori(
        item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_threshold = 1

    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[
        ["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)


def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)


def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    return list(data.loc[data["antecedents"] == item_antecedents].iloc[0, :])


# ==========================================================================================================================================================================

st.text("")
st.text("")

if type(data) != type("No result!"):
    st.markdown("""<p id="recommendation-1z2x">Recommendation:</p>""",
                unsafe_allow_html=True)
    st.success(
        f"Customer who buys **{item}**, also buys **{return_item_df(item)[1]}**!")


def get_apriori_results(transactions, min_support, min_confidence):

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    return frequent_itemsets, rules


def main():

    st.subheader("Association Rules Nilai Support & Confidence")
    st.write(rules)


if __name__ == "__main__":
    main()
