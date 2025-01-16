

""" here im going to dump all the functions that dont have to do with the specific project """


def format_time(seconds):

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60



    return f"{int(hours)}:{int(minutes)}:{seconds}"


if __name__ == "__main__":

    print(format_time(10000.123486))