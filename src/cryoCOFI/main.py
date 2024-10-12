from .carbon_film_detector import *
from .detector_for_dynamo import multi_mrc_processing_dynamo
import argparse
import os


def detector_for_relion():
    pass
    
def main():
    parser = argparse.ArgumentParser(description='''
    -----------------------------------
    cryoCOFI: CarbOn FIlm detector for cryo-EM images
    -----------------------------------
    ''',
    formatter_class=argparse.RawTextHelpFormatter,
    epilog='''
    -----------------------------------
    Email: zhen.victor.huang@gmail.com if you have any questions. 
    Please visit https://github.com/ZhenHuangLab/cryoCOFI for more information.
    -----------------------------------
    ''')
    subparsers = parser.add_subparsers(dest='command', help='Please specify the command to run!')

    # readmrc
    readmrc_parser = subparsers.add_parser('readmrc', help='Read MRC file and detect carbon film')
    readmrc_parser.add_argument('--input', '-i', type=str, required=True, help='Input MRC file')  
    readmrc_parser.add_argument('--lowpass', '-lp', type=int, default=200, help='Low pass filter cutoff angstrom')
    readmrc_parser.add_argument('--kernel_radius', '-kr', type=int, default=5, help='Kernel radius for bilateral filter')
    readmrc_parser.add_argument('--sigma_color', '-sc', type=float, default=10.0, help='Sigma color for bilateral filter')
    readmrc_parser.add_argument('--sigma_space', '-ss', type=float, default=10.0, help='Sigma space for bilateral filter')
    readmrc_parser.add_argument('--diameter', '-d', type=int, default=12000, help='Carbon Hole Diameter in Angstrom')
    readmrc_parser.add_argument('--map_cropping', '-mc', type=int, default=20, help='Removing edge pixels and cropping the image')
    readmrc_parser.add_argument('--dist_thr_inside_edge', '-dte', type=int, default=20, help='Distance threshold for inside edge pixels')
    readmrc_parser.add_argument('--mode_threshold', '-mt', type=float, default=0, help='Mode threshold for finding the carbon film edge')
    readmrc_parser.add_argument('--edge_quotient_threshold', '-eqt', type=float, default=6, help='Edge quotient threshold for finding the carbon film edge')
    readmrc_parser.add_argument('--show_fig', '-sf', action='store_true', default=False, help='Show figures if specified')
    readmrc_parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Show verbose information if specified')
    readmrc_parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use. Default is 0 and start from 0.')
    
    readdynamo_parser = subparsers.add_parser('readdynamo', help='Read Dynamo doc and tbl file and output a new tbl file without particles inside the carbon film')
    readdynamo_parser.add_argument('--doc_path', '-doc', type=str, required=True, help='Input Dynamo .doc file')
    readdynamo_parser.add_argument('--tbl_path', '-tbl', type=str, required=True, help='Input Dynamo .tbl file')
    readdynamo_parser.add_argument('--out_path', '-o', type=str, required=True, help='Output Dynamo .tbl file')
    readdynamo_parser.add_argument('--low_pass', '-lp', type=int, default=200, help='Low pass filter cutoff angstrom')
    readdynamo_parser.add_argument('--kernel_radius', '-kr', type=int, default=5, help='Kernel radius for bilateral filter')
    readdynamo_parser.add_argument('--sigma_color', '-sc', type=float, default=10.0, help='Sigma color for bilateral filter')
    readdynamo_parser.add_argument('--sigma_space', '-ss', type=float, default=10.0, help='Sigma space for bilateral filter')
    readdynamo_parser.add_argument('--diameter', '-d', type=int, default=12000, help='Carbon Hole Diameter in Angstrom')
    readdynamo_parser.add_argument('--map_cropping', '-mc', type=int, default=20, help='Removing edge pixels and cropping the image')
    readdynamo_parser.add_argument('--dist_thr_inside_edge', '-dte', type=int, default=20, help='Distance threshold for inside edge pixels')
    readdynamo_parser.add_argument('--mode_threshold', '-mt', type=float, default=0, help='Mode threshold for finding the carbon film edge')
    readdynamo_parser.add_argument('--edge_quotient_threshold', '-eqt', type=float, default=6, help='Edge quotient threshold for finding the carbon film edge')
    readdynamo_parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Show verbose information if specified')
    readdynamo_parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use. Default is 0 and start from 0.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.command == 'readmrc':
        detector_for_mrc(
            args.input,
            args.lowpass,
            args.kernel_radius,
            args.sigma_color,
            args.sigma_space,
            args.diameter,
            args.map_cropping,
            args.dist_thr_inside_edge,
            args.mode_threshold,
            args.edge_quotient_threshold,
            args.show_fig,
            args.verbose
        )
    elif args.command == 'readrelion':
        detector_for_relion()
    elif args.command == 'readdynamo':
        multi_mrc_processing_dynamo(
            args.doc_path,
            args.tbl_path,
            args.out_path,
            args.low_pass,
            args.kernel_radius,
            args.sigma_color,
            args.sigma_space,
            args.diameter,
            args.map_cropping,
            args.dist_thr_inside_edge,
            args.mode_threshold,
            args.edge_quotient_threshold,
            args.verbose
        )
    else:
        parser.print_help()

if __name__ == '__main__':
    main()