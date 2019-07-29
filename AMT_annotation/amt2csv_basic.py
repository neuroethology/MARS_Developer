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

def download_data( run_id ):
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
    where run_id = %s
    """ % ( run_id )
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
    col_labels = ['run_id','filename_hash','frame_number','annotation_id','annotation_time','annotation_version', \
                  'hit_id','worker_id','assignment_id','click_id','annotation_type','x','y','dt','frame_to_annotate_id','filename']

    # Each iteration, grab a bunch of rows and save them to file 
    annotations = []
    while True:
        report(".")
        results = cursor.fetchmany( db_rows_per_fetch )
        if results:
            for result in results:
                annotations.append( list( result ) )
        else:
            # No more rows
            break
    connection.close()
    return annotations, col_labels


## Command-line arguments
parser = argparse.ArgumentParser('Download and/or process AMT annotation data.')
parser.add_argument('-i', '--in', action='store', const='import.csv', default='', dest='in_file', metavar='in_file', nargs='?', help='Input file. Will import from CSV instead of downloading data if given.')
parser.add_argument('-o', '--out', action='store', const='top_miniscope_annotations.csv', default='', dest='out_file', metavar='out_file', nargs='?', help='Output file. Will export results to CSV if given.')
parser.add_argument('-d', '--detect', action='store_true', default=False, dest='PARAM_DETRLSWAP', help='Detect R/L swaps. Only valid for top view datasets.')
parser.add_argument('-f', '--fix', action='store_true', default=False, dest='PARAM_FIXRLSWAP', help='Fix R/L swaps (will also detect if given). Only valid for top view datasets.')
parser.add_argument('-r', '--run', action='store', default=5, type=int, dest='RUN_ID', metavar='run_id', help='Run ID to be used.')
parser.add_argument('-a', '--analyze', action='store_true', default=False, dest='PARAM_ANALYZE', help='Analyze the annotations and produce plots.')
args = parser.parse_args()

## Parameters
# Number of annotation types (clicks)
RUN_ID = args.RUN_ID

# What to do?
PARAM_DETRLSWAP = args.PARAM_DETRLSWAP
PARAM_FIXRLSWAP = args.PARAM_FIXRLSWAP
PARAM_ANALYZE = args.PARAM_ANALYZE
PARAM_EXPORTCSV = (args.out_file != '') # Export processed data to CSV
PARAM_DOWNLOAD = (args.in_file == '')  # Download data from database instead of importing from CSV

# Importing and exporting
csv_delimiter = ";" # The field separator
csv_quote_fields = False # Should we wrap fields in quotes?
db_rows_per_fetch = 500 # Number of rows to fetch from DB per batch

## Download or import data
if PARAM_DOWNLOAD == True:
    report("Downloading data.")
    annotations, col_labels = download_data( RUN_ID )
else:
    report("Importing data from csv file...")
    annotations = []
    with open(args.in_file, 'rb') as f:
        reader = csv.reader( f, delimiter=csv_delimiter, quotechar='"' )
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

### Fix data if necessary
## Flag and fix errors if appropriate
if (PARAM_DETRLSWAP == True) or (PARAM_FIXRLSWAP == True):
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
    comment = 6
    hit_id = 7
    worker_id = 8
    assignment_id = 9
    click_id = 10
    annotation_type = 11
    x = 12
    y = 13
    dt = 14
    frame_to_annotate_id = 15
    filename = 16
    script_notes = 17

    report("Checking annotations.")
    errors = 0
    frame_errors = 0
    annotations_processed = []

    col_labels += ['script_notes']

    # Check all annotations
    a_first = 0
    a_last = a_first
    report_interval = db_rows_per_fetch/(AT_W_BASE + AT_TTIP + 1)*(AT_W_BASE + AT_TTIP + 1)
    while a_last != len(annotations):
        # Report progress
        if (a_first % report_interval) == 0:
            report(".")

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
            annotation[AT_W_BASE + AT_REAR][script_notes] += "RL_SWAP,"
            annotation[AT_W_BASE + AT_LEAR][script_notes] += "RL_SWAP,"
            if PARAM_FIXRLSWAP == True:
                annotation[AT_W_BASE + AT_LEAR][x], annotation[AT_W_BASE + AT_REAR][x] = annotation[AT_W_BASE + AT_REAR][x], annotation[AT_W_BASE + AT_LEAR][x]
                annotation[AT_W_BASE + AT_LEAR][y], annotation[AT_W_BASE + AT_REAR][y] = annotation[AT_W_BASE + AT_REAR][y], annotation[AT_W_BASE + AT_LEAR][y]

        if np.cross( b_hn, b_rle )[2] > 0:
            errors += 1
            frame_error = True
            # Black R/L ear error
            annotation[AT_B_BASE + AT_REAR][script_notes] += "RL_SWAP,"
            annotation[AT_B_BASE + AT_LEAR][script_notes] += "RL_SWAP,"
            if PARAM_FIXRLSWAP == True:
                annotation[AT_B_BASE + AT_LEAR][x], annotation[AT_B_BASE + AT_REAR][x] = annotation[AT_B_BASE + AT_REAR][x], annotation[AT_B_BASE + AT_LEAR][x]
                annotation[AT_B_BASE + AT_LEAR][y], annotation[AT_B_BASE + AT_REAR][y] = annotation[AT_B_BASE + AT_REAR][y], annotation[AT_B_BASE + AT_LEAR][y]

        if np.cross( w_nt, w_rlb )[2] > 0:
            errors += 1
            frame_error = True
            # White R/L body error
            annotation[AT_W_BASE + AT_RBDY][script_notes] += "RL_SWAP,"
            annotation[AT_W_BASE + AT_LBDY][script_notes] += "RL_SWAP,"
            if PARAM_FIXRLSWAP == True:
                annotation[AT_W_BASE + AT_LBDY][x], annotation[AT_W_BASE + AT_RBDY][x] = annotation[AT_W_BASE + AT_RBDY][x], annotation[AT_W_BASE + AT_LBDY][x]
                annotation[AT_W_BASE + AT_LBDY][y], annotation[AT_W_BASE + AT_RBDY][y] = annotation[AT_W_BASE + AT_RBDY][y], annotation[AT_W_BASE + AT_LBDY][y]

        if np.cross( b_nt, b_rlb )[2] > 0:
            errors += 1
            frame_error = True
            # Black R/L body error
            annotation[AT_B_BASE + AT_RBDY][script_notes] += "RL_SWAP,"
            annotation[AT_B_BASE + AT_LBDY][script_notes] += "RL_SWAP,"
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
if PARAM_EXPORTCSV == True:
    report("Exporting CSV...")
    if csv_quote_fields:
        quote_type = csv.QUOTE_ALL
    else:
        quote_type = csv.QUOTE_NONE
    with open(args.out_file, 'wb') as f:
        writer = csv.writer(f, delimiter = csv_delimiter, quotechar = '', quoting = quote_type, escapechar='\\')
        writer.writerow(col_labels)
        for annotation in annotations:
            writer.writerow(annotation)
    report("done.\n")
    print "\tCSV file saved as " + args.out_file


## Compute metrics
if PARAM_ANALYZE == True:
    report("Computing metrics...\n")
    report("\tConverting values to appropriate formats...")
    annotations_df = pandas.DataFrame( data = np.array( annotations ), columns = col_labels )
    annotations_df[['run_id','frame_number','annotation_id','click_id','dt','frame_to_annotate_id']] = \
        annotations_df[['run_id','frame_number','annotation_id','click_id','dt','frame_to_annotate_id']].apply(pandas.to_numeric)
    annotations_df[['x','y']] = annotations_df[['x','y']].astype(str).astype(float)
    annotations_df['annotation_time'] = annotations_df['annotation_time'].apply(pandas.to_datetime)
    annotations_df['dt'] = annotations_df['dt'].divide(1000.0) # Convert times to seconds
    N_ANNOTATION_TYPES = len( annotations_df['annotation_type'].unique() ) # Find number of annotation types
    report("done.\n")

    # Time per picture (by worker)
    report("\tTime per picture...")
    worker_times = annotations_df[['worker_id', 'filename_hash', 'dt']].groupby(['worker_id', 'filename_hash'], as_index=False).sum()
    report("done.\n")

    # Time learning curve
    report("\tTime learning curve...")
    avg_by_order = []
    worker_times_byworker = worker_times.groupby('worker_id')
    for order in range(worker_times_byworker.size().max()):
        avg_by_order.append([ order, worker_times_byworker['dt'].nth(order).mean()])
    time_learning_curve_df = pandas.DataFrame( data = avg_by_order, columns = ['order', 'time'] )
    report("done.\n")

    # Annotation accuracy from median
    report("\tAnnotation accuracy...")
    annotations_df['centroid_x'] = annotations_df['x'].groupby([annotations_df['filename_hash'], annotations_df['annotation_type']]).transform('mean')
    annotations_df['centroid_y'] = annotations_df['y'].groupby([annotations_df['filename_hash'], annotations_df['annotation_type']]).transform('mean')
    annotations_df['sq_error'] = (annotations_df['x'] - annotations_df['centroid_x'])**2 + (annotations_df['y'] - annotations_df['centroid_y'])**2
    report("done.\n")

    # Frames per user
    report("\tFrames per user...")
    frames_per_user = annotations_df['worker_id'].value_counts(sort=False) / ( N_ANNOTATION_TYPES )
    report("done.\n")

    ## Plot metrics
    report("Plotting data...\n")
    # Time to completion
    report("\tTime to completion...")
    annotations_df.plot( x='annotation_time', y='annotation_id', x_compat=True )
    plt.title('Time to completion')
    plt.xlabel('Annotation time')
    plt.ylabel('Progress')
    plt.savefig('completion_time.png')
    report("done.\n")

    # Time per point
    report("\tTime per point...")
    plt.clf()
    annotations_df[['annotation_type','dt']].boxplot( by='annotation_type' )
    plt.yscale('log')
    plt.title('Time per point')
    plt.xlabel('Annotation type')
    plt.ylabel('Time [s]')
    plt.savefig('point_time_bytype.png')
    report("done.\n")

    # Average time per picture by worker
    report("\tAverage time per picture...")
    plt.clf()
    worker_times[['worker_id','dt']].boxplot( by='worker_id' )
    plt.yscale('log')
    plt.title('Time per picture by worker')
    plt.xlabel('Worker ID')
    plt.ylabel('Time [s]')
    plt.savefig('frame_time_byworker.png')
    report("done.\n")

    # Time learning curve
    report("\tTime learning curve...")
    plt.clf()
    time_learning_curve_df.plot( x='order', y='time' )
    plt.title('Time learning curve')
    plt.xlabel('Picture order')
    plt.ylabel('Time [s]')
    plt.savefig('learning_curve_time.png')
    report("done.\n")

    # Annotation error from median
    report("\tAnnotation error...")
    plt.clf()
    bins = np.logspace(-6, -1, 20)
    #bins = np.array([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1])
    n_plot_cols = 3
    n_plot_rows = int( math.ceil( float( N_ANNOTATION_TYPES ) / n_plot_cols ) )
    f, ax = plt.subplots(n_plot_rows, n_plot_cols, sharex=True, sharey=True)
    aid = 0
    for i in range(n_plot_rows):
        for j in range(n_plot_cols):
            if aid < N_ANNOTATION_TYPES:
                ax[i, j].set_xscale('log')
                ax[i, j].set_xlim((0.000001, 1))
            else:
                f.delaxes( ax.flatten()[aid] )
            aid += 1
    f.suptitle('Annotation accuracy')
    f.text(0.5, 0.04, 'Squared distance from median', ha='center', va='center')
    f.text(0.02, 0.5, 'Count [K]', ha='center', va='center', rotation='vertical')

    annotations_df[['annotation_type','sq_error']].hist(by='annotation_type', bins=bins, ax=ax.flatten()[0:N_ANNOTATION_TYPES])
    ymin, ymax = ax[0, 0].get_ylim()
    yticks = np.linspace(ymin, ymax, 4)
    for i in range(n_plot_rows):
        ax[i, 0].set_yticks(yticks)
        ax[i, 0].set_yticklabels([str(tick/1000) for tick in yticks])

    plt.savefig('accuracy_bytype.png')
    report("done.\n")

    # Frames per user (bins)
    report("\tFrames per user...")
    plt.clf()
    bins = np.logspace(0, 4, 20)
    #bins = np.array([1, 10, 100, 1000, 10000])
    frames_per_user.hist(log=True, bins=bins)
    plt.xscale('log')
    plt.xlim( (1, 10000))
    plt.title('Frames per user')
    plt.ylabel('Count')
    plt.xlabel('Number of frames')
    plt.savefig('frames_per_user.png')
    report("done.\n")

    # Accuracy vs time
    report("\tAccuracy vs time...")
    plt.clf()
    data_xfiltered = annotations_df[annotations_df['dt']>0]
    data_filtered = data_xfiltered[data_xfiltered['sq_error']>0]
    report ( "filtered " + str( len(annotations_df) - len(data_xfiltered) ) + " unlikely dt's and " + str( len(data_xfiltered) - len(data_filtered) ) + " unlikely distances..." )
    p = data_filtered.plot.hexbin( x='dt', y='sq_error', xscale='log', yscale='log', bins='log', extent=(-3, 4, -12,0), cmap='hot' )
    plt.title('Accuracy versus time')
    plt.xlabel('Time [s]')
    plt.ylabel('Squared distance from median')
    p.get_figure().text(0.90, 0.5, 'log(N)', ha='center', va='center', rotation='vertical')
    plt.savefig('accuracy_vs_time.png')

    '''
    # Plot each accuracy by time in a big matrix
    plt.clf()
    f, ax = plt.subplots(6,3, sharex=True, sharey=True)
    f.suptitle('Accuracy versus time')
    f.text(0.5, 0.04, 'Time [s]', ha='center', va='center')
    f.text(0.02, 0.5, 'Squared distance from median', ha='center', va='center', rotation='vertical')
    for i in range(6):
        for j in range(3):
            typestr = annotation_types[i*3+j]
            print typestr
            data_filtered[data_filtered['annotation_type'] == typestr].plot.hexbin( x='dt', y='sq_error', xscale='log', yscale='log', bins='log', extent=(-3, 4, -12,0), cmap='hot', ax = ax[i,j] )
    plt.savefig('accuracy_vs_time_bytype.png')
    '''
    report("done.\n")

    # Accuracy vs time, linear
    report("\tAccuracy vs time (linear)...")
    plt.clf()
    report ( "filtered " + str( len(annotations_df) - len(data_xfiltered) ) + " unlikely dt's and " + str( len(data_xfiltered) - len(data_filtered) ) + " unlikely distances..." )
    p = data_filtered.plot.hexbin( x='dt', y='sq_error', xscale='linear', yscale='linear', bins='log', extent=(0.1, 3, 0, 0.15), cmap='hot' )
    plt.title('Accuracy versus time')
    plt.xlabel('Time [s]')
    plt.ylabel('Squared distance from median')
    p.get_figure().text(0.90, 0.5, 'log(N)', ha='center', va='center', rotation='vertical')
    plt.savefig('accuracy_vs_time_lin.png')
    report("done.\n")

