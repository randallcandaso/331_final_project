from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np


def main():
    teams = pd.read_csv('Team Stats.csv', index_col=0)
    winners = pd.read_csv('SB Winners.csv')
    new = preprocessing(teams, winners)
    make_fig_1(new)
    make_fig_2(new)
    make_fig_3(winners)
    plt.show()


def preprocessing(teams, winners):
    teams['Won'] = 0

    for i in winners.index:
        teams.loc[(teams['Season'] == winners.loc[i, 'Season']) & (teams['Team'] == winners.loc[i, 'Winner']), 'Won'] = 1
    new = teams.copy()

    new['Pass.TD'] = np.log1p(new['Pass.TD'])
    new['Rush.TD'] = np.log1p(new['Rush.TD'])
    new['DPly'] = np.log1p(new['DPly'])
    new['Forced.TO'] = np.log1p(new['Forced.TO'])
    return new


def log_reg_create(df):
    feature_columns = ['Season', 'Team', "Pass.Yds", "Pass.TD", "Opp.Pass.Yds", "Opp.Pass.TD"]

    X = df[feature_columns].copy()
    y = df['Won'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.iloc[:, 2:])
    X_test_scaled = scaler.transform(X_test.iloc[:, 2:])

    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    y_probs = model.predict_proba(X_test_scaled)[:, 1]

    check = pd.concat([X_test, y_test], axis=1)
    check['probs'] = y_probs

    return check


def get_ols_parameters(x, y):
    """

    :param x: independent variable
    :param y: independent variable

    This function calculates a OLS model and returns the slope, intercept, r^2, and p_val
    values from the model.

    """
    const = sm.add_constant(x)
    model = sm.OLS(y, const)
    results = model.fit()
    slope = results.params.iloc[1]
    inter = results.params.iloc[0]
    r_sq = results.rsquared
    p_val = results.pvalues.iloc[1]
    return [slope, inter, r_sq, p_val]


def make_fig_1(df):
    metrics = ["Pass.Yds", "Pass.TD", "Opp.Pass.Yds", "Opp.Pass.TD"]
    titles = ["Pass Yards vs W-L%", "Pass TDs (transformed) vs W-L%", "Opponent Pass Yards vs W-L%", "Opponent Pass TDs vs W-L%"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        data_won_true = df[df["Won"] == 1]
        data_won_false = df[df["Won"] == 0]

        ax.scatter(data_won_false[metric], data_won_false["W-L%"], color="red", label="SB: False", s=30,
                    alpha=0.7)
        ax.scatter(data_won_true[metric], data_won_true["W-L%"], color="green", label="SB: True", s=30,
                    alpha=0.7)
        
        params = get_ols_parameters(df.loc[:, metric], df.loc[:, "W-L%"])
        line = params[0] * df.loc[:, metric] + params[1]
        ax.plot(df.loc[:, metric], line, color="blue", linestyle="dashed", linewidth=2, label="Linear Fit")

        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel(metrics[i], fontsize=10)
        ax.set_ylabel("Win Percentage", fontsize=10)
        ax.set_ylim(0, None)

    axes[0].legend(title="Super Bowl Winners", loc="best")
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.85)
    plt.suptitle("Passing Stats to Win Percentage Linear Model", fontsize=16, y=0.95)


def make_fig_2(df):
    metrics = ["Pass.Yds", "Pass.TD", "Opp.Pass.Yds", "Opp.Pass.TD"]
    titles = ["Pass Yards vs W-L%", "Pass TDs (transformed) vs W-L%", "Opponent Pass Yards vs W-L%", "Opponent Pass TDs vs W-L%"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    reg_df = log_reg_create(df)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        correct_pos = reg_df[(reg_df['Won'] == 1) & (reg_df['probs'] >= 0.5)]
        false_pos = reg_df[(reg_df['Won'] == 0) & (reg_df['probs'] >= 0.5)]
        false_neg = reg_df[(reg_df['Won'] == 1) & (reg_df['probs'] < 0.5)]
        correct_neg = reg_df[(reg_df['Won'] == 0) & (reg_df['probs'] < 0.5)]

        ax.scatter(reg_df[metric], reg_df["probs"], color="cyan", label="False Positives", s=30, alpha=0.7)
        ax.scatter(correct_pos[metric], correct_pos["probs"], color="green", label="True Positives", s=30, alpha=0.7)
        ax.scatter(false_neg[metric], false_neg["probs"], color="red", label="False Negatives", s=30, alpha=0.7)
        ax.scatter(correct_neg[metric], correct_neg["probs"], color="orange", label="True Negatives", s=30, alpha=0.7)

        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel(metrics[i], fontsize=10)
        ax.set_ylabel("Won Probability", fontsize=10)
        ax.set_ylim(0, None)

    axes[0].legend(title="Super Bowl Winners", loc="best")
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.85)
    plt.suptitle("Passing Stats to Super Bowl Logistic Model", fontsize=16, y=0.95)


def make_fig_3(winners):
    winners['Winner'] = winners['Winner'].replace('STL', 'LAR')
    counts = winners['Winner'].value_counts()

    colors = {'PIT': '#101820', 'NWE': '#002244', 'DAL': '#003594', 'KAN': '#E31837', 'NYG': '#0B2265', 'SFO': '#AA0000', 'WAS': '#773141',
              'BAL': '#241773', 'DEN': '#FB4F14', 'OAK': '#A5ACAF', 'GNB': '#FFB612', 'IND': '#002C5F', 'NOR': '#D3BC8D', 
              'SEA': '#69BE28', 'LAR': '#003594', 'MIA': '#008E97', 'TAM': '#D50A0A', 'PHI': '#004C54', 'CHI': '#C83803', 'STL': '#003594'}

    bar_colors = [colors.get(team, '#000000') for team in counts.index]

    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts, color=bar_colors)
    plt.ylabel('# of SB\'s', fontsize=16)
    plt.xlabel('Team', fontsize=16)
    plt.title('Teams with Super Bowls', fontsize=20)


main()