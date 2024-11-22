import cryoCOFI
from cryoCOFI.carbon_film_detector import *
from cryoCOFI.detector_for_dynamo import multi_mrc_processing_dynamo
from cryoCOFI.detector_for_cryosparc import multi_mrc_processing_cryosparc
from cryoCOFI.detector_for_relion import *
import argparse
import os
import sys
import setproctitle

def main():
    # set process name
    setproctitle.setproctitle('cryoCOFI')
    parser = argparse.ArgumentParser(description=f'''
    -----------------------------------
    cryoCOFI: CarbOn FIlm detector for cryo-EM images
    version {cryoCOFI.__version__}
    -----------------------------------
    ''',
    formatter_class=argparse.RawTextHelpFormatter,
    epilog='''
    -----------------------------------
    Finished at Sai Li Lab in Tsinghua University. 2024-10
    Email: zhen.victor.huang@gmail.com if you have any questions. 
    Please visit https://github.com/ZhenHuangLab/cryoCOFI for more information.
    -----------------------------------
    ''')
    subparsers = parser.add_subparsers(dest='command', help='Please specify the command to run!')
    readmrc_parser = subparsers.add_parser('readmrc', help='Read MRC file and detect carbon film')
    readdynamo_parser = subparsers.add_parser('readdynamo', help='Read Dynamo doc and tbl file and output a new tbl file without particles inside the carbon film')
    readcs_parser = subparsers.add_parser('readcs', help='Read CryoSPARC cs file and output the new cs file')
    readrelion_parser = subparsers.add_parser('readrelion', help='Read Relion star file and output the new star file')

    # Add arguments to subparsers
    readmrc_args(readmrc_parser)
    readcs_args(readcs_parser)
    readdynamo_args(readdynamo_parser)
    readrelion_args(readrelion_parser)

    try:
        args = parser.parse_args()
        
        if args.command == 'readmrc':
            handle_readmrc(args)
        elif args.command == 'readcs':
            handle_readcs(args)
        elif args.command == 'readdynamo':
            handle_readdynamo(args)
        elif args.command == 'readrelion':
            handle_readrelion(args)
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
        elif args.command == 'readrelion':
            readrelion_parser.print_help()
        else:
            parser.print_help()
        sys.exit(1)

def readmrc_args(parser):
    parser.add_argument('--input', '-i', type=str, required=True, help='[Required] Input MRC file')
    parser.add_argument('--low_pass', '-lp', type=int, default=200, help='Low pass filter cutoff angstrom. Default is 200.')
    parser.add_argument('--detector_type', '-dt', type=str, default='bicanny', help='''Specify the detector type: bicanny or canny. Default is bicanny. 
                                    For tomograms, bicanny is recommended; for SPA, canny is recommended.''')
    parser.add_argument('--kernel_radius', '-kr', type=int, default=5, help='Kernel radius for bilateral filter. Default is 5.')
    parser.add_argument('--sigma_color', '-sc', type=float, default=10.0, help='Sigma color for bilateral filter. Default is 10.0.')
    parser.add_argument('--sigma_space', '-ss', type=float, default=10.0, help='Sigma space for bilateral filter. Default is 10.0.')
    parser.add_argument('--canny_kernel', '-ck', type=int, default=2, help='Canny kernel size for edge detection. Default is 2.')
    parser.add_argument('--diameter', '-d', type=int, default=12000, help='Carbon Hole Diameter in Angstrom. Default is 12000.')
    parser.add_argument('--map_cropping', '-mc', type=int, default=20, help='Removing edge pixels and cropping the image. Default is 20 px.')
    parser.add_argument('--dist_thr_inside_edge', '-dte', type=int, default=30, help='Distance threshold for inside edge pixels. Default is 30 px.')
    parser.add_argument('--mode_threshold', '-mt', type=float, default=0, help='Mode threshold for finding the carbon film edge. Default is 0.')
    parser.add_argument('--edge_quotient_threshold', '-eqt', type=float, required=True, help='[Required] Edge quotient threshold for finding the carbon film edge. Please specify it dataset by dataset.')
    parser.add_argument('--show_fig', '-sf', action='store_true', default=True, help='Show figures if specified. Default is True.')
    parser.add_argument('--verbose', '-v', action='store_true', default=True, help='Show verbose information if specified. Default is True.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use. Default is 0 and start from 0.')

def readcs_args(parser):
    parser.add_argument('--cs_path', '-i', type=str, required=True, help='[Required] Input CryoSPARC .cs file')
    parser.add_argument('--out_path', '-o', type=str, required=True, help='[Required] Output CryoSPARC .cs file')
    parser.add_argument('--low_pass', '-lp', type=int, default=300, help='Low pass filter cutoff angstrom. Default is 300.')
    parser.add_argument('--detector_type', '-dt', type=str, default='canny', help='''Specify the detector type: bicanny or canny. Default is canny. 
                                For tomograms, bicanny is recommended; for SPA micrographs, canny is recommended.''')
    parser.add_argument('--kernel_radius', '-kr', type=int, default=5, help='Kernel radius for bilateral filter. Default is 5.')
    parser.add_argument('--sigma_color', '-sc', type=float, default=10.0, help='Sigma color for bilateral filter. Default is 10.0.')
    parser.add_argument('--sigma_space', '-ss', type=float, default=10.0, help='Sigma space for bilateral filter. Default is 10.0.')
    parser.add_argument('--canny_kernel', '-ck', type=int, default=2, help='Canny kernel size for edge detection. Default is 2.')
    parser.add_argument('--diameter', '-d', type=int, default=12000, help='Carbon Hole Diameter in Angstrom. Default is 12000.')
    parser.add_argument('--map_cropping', '-mc', type=int, default=20, help='Removing edge pixels and cropping the image. Default is 20 px.')
    parser.add_argument('--dist_thr_inside_edge', '-dte', type=int, default=20, help='Distance threshold for inside edge pixels. Default is 20 px.')
    parser.add_argument('--mode_threshold', '-mt', type=float, default=0, help='Mode threshold for finding the carbon film edge. Default is 0.')
    parser.add_argument('--edge_quotient_threshold', '-eqt', type=float, required=True, help='[Required] Edge quotient threshold for finding the carbon film edge. Please specify it dataset by dataset.')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Show verbose information if specified')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use. Default is 0 and start from 0.')

def readdynamo_args(parser):
    parser.add_argument('--doc_path', '-doc', type=str, required=True, help='[Required] Input Dynamo .doc file')
    parser.add_argument('--tbl_path', '-tbl', type=str, required=True, help='[Required] Input Dynamo .tbl file')
    parser.add_argument('--out_path', '-o', type=str, required=True, help='[Required] Output Dynamo .tbl file')
    parser.add_argument('--low_pass', '-lp', type=int, default=200, help='Low pass filter cutoff angstrom. Default is 200.')
    parser.add_argument('--detector_type', '-dt', type=str, default='bicanny', help='''Specify the detector type: bicanny or canny. Default is bicanny. 
                                For tomograms, bicanny is recommended; for SPA, canny is recommended.''')
    parser.add_argument('--kernel_radius', '-kr', type=int, default=5, help='Kernel radius for bilateral filter. Default is 5.')
    parser.add_argument('--sigma_color', '-sc', type=float, default=10.0, help='Sigma color for bilateral filter. Default is 10.0.')
    parser.add_argument('--sigma_space', '-ss', type=float, default=10.0, help='Sigma space for bilateral filter. Default is 10.0.')
    parser.add_argument('--canny_kernel', '-ck', type=int, default=2, help='Canny kernel size for edge detection. Default is 2.')
    parser.add_argument('--diameter', '-d', type=int, default=12000, help='Carbon Hole Diameter in Angstrom. Default is 12000.')
    parser.add_argument('--map_cropping', '-mc', type=int, default=20, help='Removing edge pixels and cropping the image. Default is 20 px.')
    parser.add_argument('--dist_thr_inside_edge', '-dte', type=int, default=20, help='Distance threshold for inside edge pixels. Default is 20 px.')
    parser.add_argument('--mode_threshold', '-mt', type=float, default=0, help='Mode threshold for finding the carbon film edge. Default is 0.')
    parser.add_argument('--edge_quotient_threshold', '-eqt', type=float, required=True, help='[Required] Edge quotient threshold for finding the carbon film edge. Please specify it dataset by dataset.')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Show verbose information if specified. Default is False.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use. Default is 0 and start from 0.')

# TODO: add readrelion module
def readrelion_args(parser):
    parser.add_argument('--star_path', '-s', type=str, required=True, help='[Required] Input Relion star file')
    # parser.add_argument('--out_path', '-o', type=str, required=True, help='[Required] Output Relion star file')

def print_parameters(args, command):
    print(f"""
    -----------------------------------
    cryoCOFI version {cryoCOFI.__version__} - {command}
    -----------------------------------
    """)
    if command == 'readmrc':
        print(f"""
    Input MRC file: {args.input}
        """)
    elif command == 'readcs':
        print(f"""
    Input CryoSPARC .cs file: {args.cs_path}
    Output CryoSPARC .cs file: {args.out_path}
        """)
    elif command == 'readdynamo':
        print(f"""
    Input Dynamo .doc file: {args.doc_path}
    Input Dynamo .tbl file: {args.tbl_path}
    Output Dynamo .tbl file: {args.out_path}
        """)

    print(f"""
    Low pass filter cutoff angstrom: {args.low_pass} Angstrom
    Detector type: {args.detector_type}
    (Bicanny ONLY) Bilateral filter kernel radius: {args.kernel_radius}
    (Bicanny ONLY) Bilateral filter sigma color: {args.sigma_color}
    (Bicanny ONLY) Bilateral filter sigma space: {args.sigma_space}
    (Canny ONLY) Canny kernel size: {args.canny_kernel}
    Carbon hole diameter: {args.diameter} Angstrom
    Removing the particles inside the edge of the whole image of {args.map_cropping} px (image-cropping).
    Removing the particles inside the carbon film of {args.dist_thr_inside_edge} px if detected.
    Mode threshold for finding the carbon film edge: {args.mode_threshold}
    Edge quotient threshold for finding the carbon film edge: {args.edge_quotient_threshold}
    GPU device number: {args.gpu}

    -----------------------------------
    """)

def handle_readmrc(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # print the parameters
    print_parameters(args, 'readmrc')
    detector_for_mrc(
        args.input,
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
        args.show_fig,
        args.verbose
    )

def handle_readcs(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print_parameters(args, 'readcs')
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
    print_parameters(args, 'readdynamo')
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

# TODO: add readrelion module
def handle_readrelion(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # print_parameters(args, 'readrelion')
    # read_relion_star(args.star_path, args.out_path)
    read_relion_star(args.star_path)

if __name__ == '__main__':
    main()
