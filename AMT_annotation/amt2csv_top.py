from sqlalchemy import create_engine
import MySQLdb.cursors
import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import argparse


# Helper fcts
def report( message ):
    sys.stdout.write(message)
    sys.stdout.flush()


def export_csv(csv_exp_filename, csv_imp_filename='',
               # Detect R/L swaps
               PARAM_DETRLSWAP = True,

               # Fix the detected swaps in the dataset
               PARAM_FIXRLSWAP = True,

               # Export processed data to CSV,
               PARAM_EXPORTCSV = True,

               # Download data from database instead of from CSV
               PARAM_DOWNLOAD = True):
    ## Parameters
    # What to do?

    # Importing and exporting
    csv_delimiter = ";" # The field separator
    csv_quote_fields = True # Should we wrap fields in quotes?
    # csv_imp_filename = "../../tf_dataset/data_10k_correct/export_10k.csv" # The name of the input file
    # csv_exp_filename = "../../tf_dataset/data_10k_correct/export_10k_correct.csv" # The name of the output file
    db_rows_per_fetch = 500 # Number of rows to fetch from DB per batch

    ## Constants
    # Annotation type order
    AT_B_BASE = 0
    AT_W_BASE = 9
    AT_HEAD = 0
    AT_REAR = 1
    AT_LEAR = 2
    AT_NECK = 3
    AT_RBDY = 4
    AT_LBDY = 5
    AT_TBAS = 6
    AT_TMID = 7
    AT_TTIP = 8
    annotation_types = []
    for atype in range( AT_TTIP + 1 ):
        annotation_types.append( "B" + str(atype) )
    for atype in range( AT_TTIP + 1 ):
        annotation_types.append( "W" + str(atype) )

    # Data SQL query order
    run_id = 0
    filename_hash = 1
    frame_number = 2
    annotation_id = 3
    annotation_time = 4
    annotation_version = 5
    hit_id = 6
    worker_id = 7
    assignment_id = 8
    click_id = 9
    annotation_type = 10
    x = 11
    y = 12
    dt = 13
    frame_to_annotate_id = 14
    filename = 15
    script_notes = 16

    ## Download data - Zack's code
    if PARAM_DOWNLOAD == True:
        report("Downloading data.")
        # Connect to our AWS RDS mysql instance
        # The special cursor helps not buffer the entire resultset in memory
        connection_string = 'mysql+mysqldb://mouseprojectteam:aggression@annotations.cic4uncccpt3.us-west-2.rds.amazonaws.com:3306/annotations'
        db_engine = create_engine(connection_string, pool_recycle=3600,
                              connect_args={'cursorclass': MySQLdb.cursors.SSCursor})
        query = \
        """
        select 
            run_id,filename_hash,frame_number,annotation_id,annotation_time,annotation_version,
            hit_id,worker_id,assignment_id,click_id,annotation_type,x,y,dt,frame_to_annotate_id,filename
        from annotations_new natural join clicks natural join frames_to_annotate 
        where run_id = 3
        """
        # Make sure the query is safe
        assert not "update" in query, "Please only execute SELECT queries"
        assert not "delete" in query, "Please only execute SELECT queries"
        assert not "insert" in query, "Please only execute SELECT queries"

        # Execute the query and get a cursor
        # More magic to get it to not buffer the whole result
        connection = db_engine.raw_connection()
        cursor = connection.cursor()
        result_proxy = cursor.execute(query)

        # Hard-coded column labels
        col_labels = ["run_id","filename_hash","frame_number","annotation_id","annotation_time","annotation_version",
        "hit_id","worker_id","assignment_id","click_id","annotation_type","x","y","dt","frame_to_annotate_id","filename"]

        # Each iteration, grab a bunch of rows and save them to file
        annotations = []
        while True:
            # report(".")
            results = cursor.fetchmany( db_rows_per_fetch )
            if results:
                for result in results:
                    annotations.append( list( result ) )
            else:
                # No more rows
                break
        connection.close()
    else:
        report("Importing data from csv file...")
        annotations = []
        with open(csv_imp_filename, 'rb') as f:
            reader = csv.reader( f, delimiter=csv_delimiter )
            for row in reader:
                if row[0] != 'run_id':
                    float_row = []
                    for item in row:
                        try:
                            float_row.append(float(item))
                        except ValueError:
                            float_row.append(item)
                    annotations.append( float_row )
                else:
                    col_labels = row
    report("done.\n")
    report("\tImported " + str(len(annotations)) + " rows.\n")

    ## Flag and fix errors if appropriate
    if PARAM_DETRLSWAP:
        report("Checking annotations.")
        errors = 0
        frame_errors = 0
        annotations_processed = []

        col_labels += ["script_notes"]

        # Check all annotations
        a_first = 0
        a_last = a_first
        report_interval = db_rows_per_fetch/(AT_W_BASE + AT_TTIP + 1)*(AT_W_BASE + AT_TTIP + 1)
        while a_last != len(annotations):
            # Report progress
            # if (a_first % report_interval) == 0:
            #     report(".")

            # Get all datapoints for that annotation
            while a_last != len(annotations) and annotations[a_last][annotation_id] == annotations[a_first][annotation_id]:
                a_last += 1
            annotation = annotations[a_first : a_last]

            # Check that there are enough points for that annotation
            not_enough = ( len(annotation) != AT_W_BASE + AT_TTIP + 1 )
            for i in range(len(annotation)):
                if not_enough:
                    annotation[i].append('NOT_ENOUGH') # Script notes
                else:
                    annotation[i].append('') # Script notes

            if not_enough:
                report("INFO: Annotation missing, sequence id =" + str(a_first))
                a_first = a_last
                continue

            # Compute vectors
            w_hn  = np.array( [ annotation[AT_W_BASE + AT_NECK][x] - annotation[AT_W_BASE + AT_HEAD][x],
                                annotation[AT_W_BASE + AT_NECK][y] - annotation[AT_W_BASE + AT_HEAD][y],
                                0 ] )
            b_hn  = np.array( [ annotation[AT_B_BASE + AT_NECK][x] - annotation[AT_B_BASE + AT_HEAD][x],
                                annotation[AT_B_BASE + AT_NECK][y] - annotation[AT_B_BASE + AT_HEAD][y],
                                0 ] )
            w_rle = np.array( [ annotation[AT_W_BASE + AT_LEAR][x] - annotation[AT_W_BASE + AT_REAR][x],
                                annotation[AT_W_BASE + AT_LEAR][y] - annotation[AT_W_BASE + AT_REAR][y],
                                0 ] )
            b_rle = np.array( [ annotation[AT_B_BASE + AT_LEAR][x] - annotation[AT_B_BASE + AT_REAR][x],
                                annotation[AT_B_BASE + AT_LEAR][y] - annotation[AT_B_BASE + AT_REAR][y],
                                0 ] )
            w_nt  = np.array( [ annotation[AT_W_BASE + AT_TBAS][x] - annotation[AT_W_BASE + AT_NECK][x],
                                annotation[AT_W_BASE + AT_TBAS][y] - annotation[AT_W_BASE + AT_NECK][y],
                                0 ] )
            b_nt  = np.array( [ annotation[AT_B_BASE + AT_TBAS][x] - annotation[AT_B_BASE + AT_NECK][x],
                                annotation[AT_B_BASE + AT_TBAS][y] - annotation[AT_B_BASE + AT_NECK][y],
                                0 ] )
            w_rlb = np.array( [ annotation[AT_W_BASE + AT_LBDY][x] - annotation[AT_W_BASE + AT_RBDY][x],
                                annotation[AT_W_BASE + AT_LBDY][y] - annotation[AT_W_BASE + AT_RBDY][y],
                                0 ] )
            b_rlb = np.array( [ annotation[AT_B_BASE + AT_LBDY][x] - annotation[AT_B_BASE + AT_RBDY][x],
                                annotation[AT_B_BASE + AT_LBDY][y] - annotation[AT_B_BASE + AT_RBDY][y],
                                0 ] )

            # Detect errors and report/fix as necessary
            frame_error = False
            if np.cross( w_hn, w_rle )[2] > 0:
                errors += 1
                frame_error = True
                # White R/L ear error
                annotation[AT_W_BASE + AT_REAR][script_notes] += "RL_SWAP"
                annotation[AT_W_BASE + AT_LEAR][script_notes] += "RL_SWAP"
                if PARAM_FIXRLSWAP == True:
                    annotation[AT_W_BASE + AT_LEAR][x], annotation[AT_W_BASE + AT_REAR][x] = annotation[AT_W_BASE + AT_REAR][x], annotation[AT_W_BASE + AT_LEAR][x]
                    annotation[AT_W_BASE + AT_LEAR][y], annotation[AT_W_BASE + AT_REAR][y] = annotation[AT_W_BASE + AT_REAR][y], annotation[AT_W_BASE + AT_LEAR][y]

            if np.cross( b_hn, b_rle )[2] > 0:
                errors += 1
                frame_error = True
                # Black R/L ear error
                annotation[AT_B_BASE + AT_REAR][script_notes] += "RL_SWAP"
                annotation[AT_B_BASE + AT_LEAR][script_notes] += "RL_SWAP"
                if PARAM_FIXRLSWAP == True:
                    annotation[AT_B_BASE + AT_LEAR][x], annotation[AT_B_BASE + AT_REAR][x] = annotation[AT_B_BASE + AT_REAR][x], annotation[AT_B_BASE + AT_LEAR][x]
                    annotation[AT_B_BASE + AT_LEAR][y], annotation[AT_B_BASE + AT_REAR][y] = annotation[AT_B_BASE + AT_REAR][y], annotation[AT_B_BASE + AT_LEAR][y]

            if np.cross( w_nt, w_rlb )[2] > 0:
                errors += 1
                frame_error = True
                # White R/L body error
                annotation[AT_W_BASE + AT_RBDY][script_notes] += "RL_SWAP"
                annotation[AT_W_BASE + AT_LBDY][script_notes] += "RL_SWAP"
                if PARAM_FIXRLSWAP == True:
                    annotation[AT_W_BASE + AT_LBDY][x], annotation[AT_W_BASE + AT_RBDY][x] = annotation[AT_W_BASE + AT_RBDY][x], annotation[AT_W_BASE + AT_LBDY][x]
                    annotation[AT_W_BASE + AT_LBDY][y], annotation[AT_W_BASE + AT_RBDY][y] = annotation[AT_W_BASE + AT_RBDY][y], annotation[AT_W_BASE + AT_LBDY][y]

            if np.cross( b_nt, b_rlb )[2] > 0:
                errors += 1
                frame_error = True
                # Black R/L body error
                annotation[AT_B_BASE + AT_RBDY][script_notes] += "RL_SWAP"
                annotation[AT_B_BASE + AT_LBDY][script_notes] += "RL_SWAP"
                if PARAM_FIXRLSWAP == True:
                    annotation[AT_B_BASE + AT_LBDY][x], annotation[AT_B_BASE + AT_RBDY][x] = annotation[AT_B_BASE + AT_RBDY][x], annotation[AT_B_BASE + AT_LBDY][x]
                    annotation[AT_B_BASE + AT_LBDY][y], annotation[AT_B_BASE + AT_RBDY][y] = annotation[AT_B_BASE + AT_RBDY][y], annotation[AT_B_BASE + AT_LBDY][y]
            annotations_processed += annotation

            if frame_error:
                frame_errors += 1

            a_first = a_last

        annotations = annotations_processed

        report("done.\n")
        n_frames = len(annotations)/(AT_W_BASE + AT_TTIP + 1)
        print "\tVerification complete. Detected", str(errors), "errors on", str(frame_errors), "frames (", str(100*float(errors)/float(n_frames*4)) ,"/", str(100*float(frame_errors)/float(n_frames)) ,"% )."

    ## Output CSV file if necessary
    csv_quote_fields = False # Should we wrap fields in quotes?
    if PARAM_EXPORTCSV == True:
        report("Exporting CSV...")
        if csv_quote_fields:
            quote_type = csv.QUOTE_ALL
            quote_char ='"'
        else:
            quote_type = csv.QUOTE_NONE
            quote_char = ''
        with open(csv_exp_filename, 'wb') as f:
            writer = csv.writer(f, delimiter = csv_delimiter, quotechar = quote_char, quoting = quote_type)
            writer.writerow(col_labels)
            for annotation in annotations:
                writer.writerow(annotation)
        report("done.\n")
        print "\tCSV file saved as" + csv_exp_filename



def parse_args():

    parser = argparse.ArgumentParser(description='Export AMT annotations to CSV')

    parser.add_argument('--save_file', dest='csv_exp_filename',
                        help='paths where to save csv file', type=str, required=True)
    parser.add_argument('--import_file', dest='csv_imp_filename',
                        help='paths csv to import', type=str, required=False)
    parser.add_argument('--lr_swap', dest='PARAM_DETRLSWAP',
                        help='Detect R/L swaps', type=bool, required=False, default=True)
    parser.add_argument('--fix_swap', dest='PARAM_FIXRLSWAP',
                        help='Fix R/L swaps', type=bool, required=False, default=True)
    parser.add_argument('--exp_csv', dest='PARAM_EXPORTCSV',
                        help='Export csv', type=bool, required=False, default=True)
    parser.add_argument('--download', dest='PARAM_DOWNLOAD',
                        help='Download annotations froms server', type=bool, required=False, default=True)


def main():
  args = parse_args()
  export_csv(args.csv_exp_filename, args.csv_imp_filename,
             args.PARAM_DETRLSWAP, args.PARAM_FIXRLSWAP,
             args.PARAM_EXPORTCSV, args.PARAM_DOWNLOAD)



if __name__ == '__main__':
  main()