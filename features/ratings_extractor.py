import pandas as pd


def categorise(x):
    if x in range(1, 26):
        return ["CSQ", "Orig"]
    elif x in range(26, 51):
        return ["CSQ", "MaMa"]
    elif x in range(51, 76):
        return ["CSQ", "CoRe"]
    elif x in range(76, 101):
        return ["CSQ", "MVAE"]
    elif x in range(101, 126):
        return ["CSQ", "MuTr"]
    elif x in range(126, 151):
        return ["CSQ", "BeAf"]
    elif x in range(151, 176):
        return ["CPI", "Orig"]
    elif x in range(176, 201):
        return ["CPI", "MVAE"]
    elif x in range(201, 226):
        return ["CPI", "MuTr"]
    elif x in range(226, 251):
        return ["CPI", "LiTr"]
    else:
        raise ValueError("Invalid excerpt ID.")


def get_ratings(split=True):
    trust = [10020, 10021, 10023, 10027, 10028, 10029, 10030, 10031, 10033, 10035, 10037, 10039, 10041, 10043,
             10045, 10046, 10049, 10050, 10051, 10053, 10056, 10057, 10059, 10060, 10063, 10065, 10068, 10070,
             10071, 10072, 10073, 10074, 10075, 10077, 10082, 10083, 10087, 10088, 10089, 10091, 10095]
    submissions = pd.read_csv("submission.csv", sep="\t")
    submissions = submissions.loc[submissions["participant_id"].isin(trust)]

    if split:
        result = {"csq_ss": [], "csq_ap": [], "csq_re": [], "csq_me": [], "csq_ha": [], "csq_rh": [],
                  "cpi_ss": [], "cpi_ap": [], "cpi_re": [], "cpi_me": [], "cpi_ha": [], "cpi_rh": []}
        for row in submissions.iterrows():
            row = row[1]
            if categorise(row["excerpt_id"])[0] == "csq":
                result["csq_ss"].append([row["a_rating"], categorise(row["excerpt_id"])[1]])
                result["csq_ap"].append([row["b_rating"], categorise(row["excerpt_id"])[1]])
                result["csq_re"].append([row["c_rating"], categorise(row["excerpt_id"])[1]])
                result["csq_me"].append([row["d_rating"], categorise(row["excerpt_id"])[1]])
                result["csq_ha"].append([row["e_rating"], categorise(row["excerpt_id"])[1]])
                result["csq_rh"].append([row["f_rating"], categorise(row["excerpt_id"])[1]])
            elif categorise(row["excerpt_id"])[0] == "cpi":
                result["cpi_ss"].append([row["a_rating"], categorise(row["excerpt_id"])[1]])
                result["cpi_ap"].append([row["b_rating"], categorise(row["excerpt_id"])[1]])
                result["cpi_re"].append([row["c_rating"], categorise(row["excerpt_id"])[1]])
                result["cpi_me"].append([row["d_rating"], categorise(row["excerpt_id"])[1]])
                result["cpi_ha"].append([row["e_rating"], categorise(row["excerpt_id"])[1]])
                result["cpi_rh"].append([row["f_rating"], categorise(row["excerpt_id"])[1]])
            else:
                raise ValueError("Invalid.")

        for k, v in result.items():
            pd.DataFrame(v, columns=["Rating", "Category"]).to_csv(f"{k}.csv", index=False)

    else:
        # result = {"csq": [], "cpi": []}
        result = []
        for row in submissions.iterrows():
            row = row[1]
            # result[categorise(row["excerpt_id"])[0]].append([row["a_rating"], categorise(row["excerpt_id"])[1], "Ss"])
            # result[categorise(row["excerpt_id"])[0]].append([row["b_rating"], categorise(row["excerpt_id"])[1], "Ap"])
            # result[categorise(row["excerpt_id"])[0]].append([row["c_rating"], categorise(row["excerpt_id"])[1], "Re"])
            # result[categorise(row["excerpt_id"])[0]].append([row["d_rating"], categorise(row["excerpt_id"])[1], "Me"])
            # result[categorise(row["excerpt_id"])[0]].append([row["e_rating"], categorise(row["excerpt_id"])[1], "Ha"])
            # result[categorise(row["excerpt_id"])[0]].append([row["f_rating"], categorise(row["excerpt_id"])[1], "Rh"])
            result.append([row["a_rating"], categorise(row["excerpt_id"])[1], "Ss", categorise(row["excerpt_id"])[0],
                           row["excerpt_id"]])
            result.append([row["b_rating"], categorise(row["excerpt_id"])[1], "Ap", categorise(row["excerpt_id"])[0],
                           row["excerpt_id"]])
            result.append([row["c_rating"], categorise(row["excerpt_id"])[1], "Re", categorise(row["excerpt_id"])[0],
                           row["excerpt_id"]])
            result.append([row["d_rating"], categorise(row["excerpt_id"])[1], "Me", categorise(row["excerpt_id"])[0],
                           row["excerpt_id"]])
            result.append([row["e_rating"], categorise(row["excerpt_id"])[1], "Ha", categorise(row["excerpt_id"])[0],
                           row["excerpt_id"]])
            result.append([row["f_rating"], categorise(row["excerpt_id"])[1], "Rh", categorise(row["excerpt_id"])[0],
                           row["excerpt_id"]])
        pd.DataFrame(result, columns=["Rating", "Category", "Aspect", "Part", "Name"]).to_csv("ratings.csv", index=False)
        # for k, v in result.items():
        #     pd.DataFrame(v, columns=["Rating", "Category", "Aspect"]).to_csv(f"{k}.csv", index=False)


if __name__ == '__main__':
    get_ratings(False)
