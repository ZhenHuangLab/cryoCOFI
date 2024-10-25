from .carbon_film_detector import *
from .detector_for_dynamo import multi_mrc_processing_dynamo
from .detector_for_cryosparc import multi_mrc_processing_cryosparc
import argparse
import os
import sys
import setproctitle
    
def main():
    # set process name
    setproctitle.setproctitle('cryoCOFI')
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
    readmrc_parser = subparsers.add_parser('readmrc', help='Read MRC file and detect carbon film')
    readdynamo_parser = subparsers.add_parser('readdynamo', help='Read Dynamo doc and tbl file and output a new tbl file without particles inside the carbon film')
    readcs_parser = subparsers.add_parser('readcs', help='Read CryoSPARC cs file and output the new cs file')

    # Add arguments to subparsers
    readmrc_args(readmrc_parser)
    readcs_args(readcs_parser)
    readdynamo_args(readdynamo_parser)

    try:
        args = parser.parse_args()
        
        if args.command == 'readmrc':
            handle_readmrc(args)
        elif args.command == 'readcs':
            handle_readcs(args)
        elif args.command == 'readdynamo':
            handle_readdynamo(args)
        else:
            parser.print_help()
            
    except (argparse.ArgumentError, argparse.ArgumentTypeError) as e:
        print(f"\nError: {str(e)}\n")
        if args.command == 'readmrc':
            readmrc_parser.print_help()
        elif args.command == 'readcs':
            readcs_parser.print_help()
        elif args.command == 'readdynamo':
            readdynamo_parser.print_help()
        else:
            parser.print_help()
        sys.exit(1)

def readmrc_args(parser):
    parser.add_argument('--input', '-i', type=str, required=True, help='Input MRC file')
    parser.add_argument('--lowpass', '-lp', type=int, default=200, help='Low pass filter cutoff angstrom')
    parser.add_argument('--detector_type', '-dt', type=str, default='bicanny', help='''Specify the detector type: bicanny or canny. Default is bicanny. 
                                    For tomograms, bicanny is recommended; for SPA, canny is recommended.''')
    parser.add_argument('--kernel_radius', '-kr', type=int, default=5, help='Kernel radius for bilateral filter')
    parser.add_argument('--sigma_color', '-sc', type=float, default=10.0, help='Sigma color for bilateral filter')
    parser.add_argument('--sigma_space', '-ss', type=float, default=10.0, help='Sigma space for bilateral filter')
    parser.add_argument('--canny_kernel', '-ck', type=int, default=2, help='Canny kernel size for edge detection')
    parser.add_argument('--diameter', '-d', type=int, default=12000, help='Carbon Hole Diameter in Angstrom')
    parser.add_argument('--map_cropping', '-mc', type=int, default=20, help='Removing edge pixels and cropping the image')
    parser.add_argument('--dist_thr_inside_edge', '-dte', type=int, default=20, help='Distance threshold for inside edge pixels')
    parser.add_argument('--mode_threshold', '-mt', type=float, default=0, help='Mode threshold for finding the carbon film edge')
    parser.add_argument('--edge_quotient_threshold', '-eqt', type=float, required=True, help='Edge quotient threshold for finding the carbon film edge')
    parser.add_argument('--show_fig', '-sf', action='store_true', default=False, help='Show figures if specified')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Show verbose information if specified')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use. Default is 0 and start from 0.')

def readcs_args(parser):
    parser.add_argument('--cs_path', '-i', type=str, required=True, help='Input CryoSPARC .cs file')
    parser.add_argument('--out_path', '-o', type=str, required=True, help='Output CryoSPARC .cs file')
    parser.add_argument('--low_pass', '-lp', type=int, default=300, help='Low pass filter cutoff angstrom')
    parser.add_argument('--detector_type', '-dt', type=str, default='canny', help='''Specify the detector type: bicanny or canny. Default is bicanny. 
                                For tomograms, bicanny is recommended; for SPA, canny is recommended.''')
    parser.add_argument('--kernel_radius', '-kr', type=int, default=5, help='Kernel radius for bilateral filter')
    parser.add_argument('--sigma_color', '-sc', type=float, default=10.0, help='Sigma color for bilateral filter')
    parser.add_argument('--sigma_space', '-ss', type=float, default=10.0, help='Sigma space for bilateral filter')
    parser.add_argument('--canny_kernel', '-ck', type=int, default=2, help='Canny kernel size for edge detection')
    parser.add_argument('--diameter', '-d', type=int, default=12000, help='Carbon Hole Diameter in Angstrom')
    parser.add_argument('--map_cropping', '-mc', type=int, default=20, help='Removing edge pixels and cropping the image')
    parser.add_argument('--dist_thr_inside_edge', '-dte', type=int, default=20, help='Distance threshold for inside edge pixels')
    parser.add_argument('--mode_threshold', '-mt', type=float, default=0, help='Mode threshold for finding the carbon film edge')
    parser.add_argument('--edge_quotient_threshold', '-eqt', type=float, required=True, help='Edge quotient threshold for finding the carbon film edge. Please specify it dataset by dataset.')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Show verbose information if specified')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use. Default is 0 and start from 0.')

def readdynamo_args(parser):
    parser.add_argument('--doc_path', '-doc', type=str, required=True, help='Input Dynamo .doc file')
    parser.add_argument('--tbl_path', '-tbl', type=str, required=True, help='Input Dynamo .tbl file')
    parser.add_argument('--out_path', '-o', type=str, required=True, help='Output Dynamo .tbl file')
    parser.add_argument('--low_pass', '-lp', type=int, default=200, help='Low pass filter cutoff angstrom')
    parser.add_argument('--detector_type', '-dt', type=str, default='bicanny', help='''Specify the detector type: bicanny or canny. Default is bicanny. 
                                For tomograms, bicanny is recommended; for SPA, canny is recommended.''')
    parser.add_argument('--kernel_radius', '-kr', type=int, default=5, help='Kernel radius for bilateral filter')
    parser.add_argument('--sigma_color', '-sc', type=float, default=10.0, help='Sigma color for bilateral filter')
    parser.add_argument('--sigma_space', '-ss', type=float, default=10.0, help='Sigma space for bilateral filter')
    parser.add_argument('--canny_kernel', '-ck', type=int, default=2, help='Canny kernel size for edge detection')
    parser.add_argument('--diameter', '-d', type=int, default=12000, help='Carbon Hole Diameter in Angstrom')
    parser.add_argument('--map_cropping', '-mc', type=int, default=20, help='Removing edge pixels and cropping the image')
    parser.add_argument('--dist_thr_inside_edge', '-dte', type=int, default=20, help='Distance threshold for inside edge pixels')
    parser.add_argument('--mode_threshold', '-mt', type=float, default=0, help='Mode threshold for finding the carbon film edge')
    parser.add_argument('--edge_quotient_threshold', '-eqt', type=float, default=6, help='Edge quotient threshold for finding the carbon film edge')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Show verbose information if specified')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use. Default is 0 and start from 0.')

def handle_readmrc(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    detector_for_mrc(
        args.input,
        args.lowpass,
        args.detector_type,
        args.kernel_radius,
        args.sigma_color,
        args.sigma_space,
        args.canny_kernel,
        args.diameter,
        args.map_cropping,
        args.dist_thr_inside_edge,
        args.mode_threshold,
        args.edge_quotient_threshold,
        args.show_fig,
        args.verbose
    )

def handle_readcs(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    multi_mrc_processing_cryosparc(
        args.cs_path,
        args.out_path,
        args.low_pass,
        args.detector_type,
        args.kernel_radius,
        args.sigma_color,
        args.sigma_space,
        args.canny_kernel,
        args.diameter,
        args.map_cropping,
        args.dist_thr_inside_edge,
        args.mode_threshold,
        args.edge_quotient_threshold,
        args.verbose
    )

def handle_readdynamo(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    multi_mrc_processing_dynamo(
        args.doc_path,
        args.tbl_path,
        args.out_path,
        args.low_pass,
        args.detector_type,
        args.kernel_radius,
        args.sigma_color,
        args.sigma_space,
        args.canny_kernel,
        args.diameter,
        args.map_cropping,
        args.dist_thr_inside_edge,
        args.mode_threshold,
        args.edge_quotient_threshold,
        args.verbose
    )

if __name__ == '__main__':
    main()
