import pandas as pd


def main():
    trust = [10020, 10021, 10023, 10027, 10028, 10029, 10030, 10031, 10033, 10035, 10037, 10039, 10041, 10043,
             10045, 10046, 10049, 10050, 10051, 10053, 10056, 10057, 10059, 10060, 10063, 10065, 10068, 10070,
             10071, 10072, 10073, 10074, 10075, 10077, 10082, 10083, 10087, 10088, 10089, 10091, 10095]
    submissions = pd.read_csv("participant.csv", sep="\t")
    submissions = submissions.loc[submissions["participant_id"].isin(trust)]

    result = []
    for row in submissions.iterrows():
        row = row[1]
        result.append([row["age"], row["years_musical_training"]])
    pd.DataFrame(result, columns=["age", "mt"]).to_csv("data.csv", index=False)


if __name__ == '__main__':
    main()
