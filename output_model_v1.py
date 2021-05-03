import pandas as pd
import json
import pickle
import boto3
sm = boto3.client('sagemaker')


def ml_function(data_row):
    # Importing new data, converting it to a df
    data = json.dumps(data_row)
    data = "["+data+"]"
    df_new = pd.read_json(data)
    print(df_new)
    # Loading ml model
    session = boto3.session.Session(region_name='us-east-1')
    s3client = session.client('s3')
    bucket='generated-data'
    data_key = 'model_v1.pkl'
    response = s3client.get_object(Bucket=bucket, Key=data_key)
    ml_model = response['Body'].read()
    testing = pickle.loads(ml_model)
    # Running model on new data
    predictions = testing.predict(df_new)
    # Converting output to a list
    df_new["Predictions"] = list(predictions)
    # Converting possible sudden responses to numerical
    cleanup = {"Sudden": {"Y": 1, "N": 0, "y": 1, "n": 0}}
    df_new = df_new.replace(cleanup)

    # Cleaning up predictions
    df_new.columns = df_new.columns.str.replace(" ", "_")
    df_new.columns = df_new.columns.str.replace("-", "_")
    df_new.columns = df_new.columns.str.replace("5_", "Five_")
    df_new.columns = df_new.columns.str.replace("/", "_")
    # Creating output for 3 and 5 recs, as well as the recs everyone gets
    recs_three = []
    recs_five = []
    green = []
    for row in df_new.itertuples():
        rank1 = []
        rank2 = []
        rank3 = []
        rank4 = []
        rank5 = []
        # 5-HTP
        if row.Predictions[0] == 1:
            rank1.append("5-126-[1]")
        elif row.Predictions[0] == 2:
            if row.stress < 34:
                rank2.append("5-126-[1]")
            elif row.stress > 66 and 34 < row.depression < 67:
                rank2.append("5-126-[1]")
            else:
                rank2.append("5-126-[2]")
        elif row.Predictions[0] == 3:
            rank3.append("5-126-[2]")
        elif row.Predictions[0] == 4:
            rank4.append("5-126-[1]")
        elif row.Predictions[0] == 5:
            rank5.append("5-126-[1]")
        # ASHWAGANDA
        if row.Predictions[1] == 1:
            if row.stress < 34:
                rank1.append("6-69-[5]")
            else:
                rank1.append("6-69-[9]")
        elif row.Predictions[1] == 2:
            rank2.append("6-69-[5]")
        elif row.Predictions[1] == 3:
            rank3.append("6-69-[5]")
        elif row.Predictions[1] == 4:
            rank4.append("6-69-[5]")
        elif row.Predictions[1] == 5:
            rank5.append("6-69-[5]")
        # Bacopa_300_mg_cap
        if row.Predictions[2] == 1:
            rank1.append("6-70-[5]")
        elif row.Predictions[2] == 2:
            rank2.append("6-70-[5]")
        elif row.Predictions[2] == 3:
            rank3.append("6-70-[5]")
        elif row.Predictions[2] == 4:
            rank4.append("6-70-[6]")
        elif row.Predictions[2] == 5:
            rank5.append("6-70-[5]")
        # Rhodiola_100_mg_cap
        if row.Predictions[3] == 1:
            rank1.append("6-67-[5]")
        elif row.Predictions[3] == 2:
            rank2.append("6-67-[5]")
        elif row.Predictions[3] == 3:
            rank3.append("6-67-[5]")
        elif row.Predictions[3] == 4:
            rank4.append("6-67-[5]")
        elif row.Predictions[3] == 5:
            rank5.append("6-67-[5]")
        # L_Theanine_200_mg_cap
        if row.Predictions[5] == 1:
            if row.sudden == 1:
                rank1.append("5-55-[6]")
            else:
                rank1.append("5-55-[5]")
        elif row.Predictions[5] == 2:
            rank2.append("5-55-[5]")
        elif row.Predictions[5] == 3:
            if row.stress < 34:
                rank3.append("5-55-[5]")
            else:
                rank3.append("5-55-[6]")
        elif row.Predictions[5] == 4:
            rank4.append("5-55-[6]")
        elif row.Predictions[5] == 5:
            rank5.append("5-55-[6]")
        # Phosphitidylserine_150_mg_cap
        if row.Predictions[6] == 1:
            rank1.append("5-124-[5]")
        elif row.Predictions[6] == 2:
            rank2.append("5-124-[6]")
        elif row.Predictions[6] == 3:
            rank3.append("5-124-[5]")
        elif row.Predictions[6] == 4:
            rank4.append("5-124-[5]")
        elif row.Predictions[6] == 5:
            rank5.append("5-124-[5]")
        # Relora_250_mg_cap
        if row.Predictions[7] == 1:
            rank1.append("6-125-[5]")
        elif row.Predictions[7] == 2:
            rank2.append("6-125-[10]")
        elif row.Predictions[7] == 3:
            if row.sudden == 1:
                rank3.append("6-125-[6]")
            else:
                rank3.append("6-125-[9]")
        elif row.Predictions[7] == 4:
            rank4.append("6-125-[5]")
        elif row.Predictions[7] == 5:
            rank5.append("6-125-[5]")
        # Holy_Basil_500_mg_cap
        if row.Predictions[8] == 1:
            rank1.append("6-73-[5]")
        elif row.Predictions[8] == 2:
            rank2.append("6-73-[5]")
        elif row.Predictions[8] == 3:
            rank3.append("6-73-[5]")
        elif row.Predictions[8] == 4:
            rank4.append("6-73-[9]")
        elif row.Predictions[8] == 5:
            if row.anxiety > 66 and row.depression > 66:
                rank5.append("6-73-[9]")
            else:
                rank5.append("6-73-[5]")
        # Curcumin_500_mg_caps
        if row.Predictions[9] == 1:
            rank1.append("6-72-[5]")
        elif row.Predictions[9] == 2:
            rank2.append("6-72-[6]")
        elif row.Predictions[9] == 3:
            if row.depression < 34:
                rank3.append("6-72-[9]")
            else:
                rank3.append("6-72-[6]")
        elif row.Predictions[9] == 4:
            if row.stress < 34:
                rank3.append("6-72-[9]")
            else:
                rank3.append("6-72-[6]")
        elif row.Predictions[9] == 5:
            rank5.append("6-72-[9]")
        else:
            pass
        # Vit D
        green.append("112-[4]")
        # Mag Bisglycinate
        if row.stress < 34:
            green.append("119-[2]")
        elif row.anxiety < 34:
            green.append("119-[2]")
        elif 102 < row.stress + row.anxiety + row.depression < 198:
            green.append("119-[3]")
        else:
            green.append("119-[4]")
        # Omega-3
        if row.stress + row.anxiety + row.depression < 102:
            green.append("66-[1]")
        elif 102 < row.stress + row.anxiety + row.depression < 150:
            green.append("66-[2]")
        elif 150 < row.stress + row.anxiety + row.depression < 198:
            green.append("66-[3]")
        elif 198 < row.stress + row.anxiety + row.depression < 245:
            green.append("66-[4]")
        else:
            green.append("66-[5]")
        # B6
        if row.stress + row.anxiety + row.depression < 210:
            green.append("115-[1]")
        else:
            green.append("115-[2]")

        recs_five.append(rank1 + rank2 + rank3 + rank4 + rank5)
        recs_three.append(rank1 + rank2 + rank3)

    df_new["Top_Three_Recs"] = recs_three
    df_new["Top_Five_Recs"] = recs_five
    df_new["Green_Supps"] = [green]

    # Creating column lists that separate recs based on whether they were a 5 or a 6
    topic_id_5 = []
    topic_id_6 = []
    for i in range(len(df_new)):
        five = []
        six = []
        for j in range(3):
            if not df_new["Top_Three_Recs"].iloc[i]:
                pass
            elif df_new["Top_Three_Recs"].iloc[i][j][0] == "5":
                five.append(df_new["Top_Three_Recs"].iloc[i][j][2:])
            elif df_new["Top_Three_Recs"].iloc[i][j][0] == "6":
                six.append(df_new["Top_Three_Recs"].iloc[i][j][2:])
            else:
                pass
            if j == 2:
                topic_id_5.append(five)
                topic_id_6.append(six)

    df_new["topic_id_5"] = topic_id_5
    df_new["topic_id_6"] = topic_id_6
    df_new["Supp_Recs_5"] = df_new["topic_id_5"] + df_new["Green_Supps"]

    # Updating supp recs into dict format
    updated_5 = []
    for i in df_new["Supp_Recs_5"]:
        for j in range(len(i)):
            d = dict(x.split("-") for x in i[j].split(","))
            updated_5.append(d)

    df_new['Supps_Output_5'] = [updated_5]

    updated_6 = []
    for i in df_new["topic_id_6"]:
        for j in range(len(i)):
            d = dict(x.split("-") for x in i[j].split(","))
            updated_6.append(d)

    df_new['Supps_Output_6'] = [updated_6]

    # Creating json output
    # This creates the general json format in dict form
    d = []
    for i in range(len(df_new)):
        d.append({"5": df_new["Supps_Output_5"].iloc[i], "6": df_new["Supps_Output_6"].iloc[i]})

    df_new["json_prep"] = d

    # Convert the dict into json
    json_outputs = []
    for i in range(len(df_new)):
        json_prep = json.dumps(df_new["json_prep"].iloc[i])
        json_outputs.append(json.loads(json_prep))

    # Formatting json to match desired output
    a = json_outputs
    a = str(a).replace("\'", '\"')
    a = a.replace('\"[', '[\"')
    a = a.replace(']\"', '\"]')
    df_new["JSON_Output"] = a

    this_is_what_you_want = df_new["JSON_Output"].iloc[0][1:-1]

    return this_is_what_you_want