from __future__ import division



import cozmo
import re
import sys
import time
import os
import errno
import math
import ctypes
import pygame
import asyncio
import io
import pyaudio
import webbrowser
import numpy as np
import matplotlib.pyplot as plt

from google.cloud import vision
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from six.moves import queue
from sympy import sympify,lambdify, Symbol
from wtforms import Form
from wtforms import TextAreaField
from wtforms import StringField 
from flask import Flask, request, render_template
from cozmo.util import distance_mm, speed_mmps, degrees
from time import localtime, strftime


try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Cannot    import from PIL. Do `pip3 install --user Pillow` to install")

global dir_files
global text_code



dir_files = 'files'
dir_pictures = 'pictures'
dir_results = 'results'

#x = ""



try:
    os.makedirs(dir_pictures)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(dir_files)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
try:
    os.makedirs(dir_results)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise



app = Flask(__name__)

client_vision = vision.ImageAnnotatorClient()

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

# See http://g.co/cloud/speech/docs/languages
# for a list of supported languages.

#Dutch = 'nl-NL'
#portuguese = 'pt-BR'
#en = en-US
language_code = 'nl-NL'  # a BCP-47 language tag
client = speech.SpeechClient()
config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code=language_code)
streaming_config = types.StreamingRecognitionConfig(
    config=config,
    interim_results=True)


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)
# [END audio_stream]
            
def listen_print_loop(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    global speech_rec
    speech_rec = ""
    
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            
            '''if re.search(r'\b(exit|quit|sair)\b', transcript, re.I):
                print('Exiting..')
                transcript = ""
                
                return  transcript'''
            
            speech_rec = transcript.upper()
            print("\n\nspeech_rec:",speech_rec,"\n\n")
                
            return  transcript
        
            if re.search(r'\b(resolvido|solved)\b', transcript, re.I):
                break
                
            #else:
            
            #print("transcript:",transcript,"\n")
            #print("overwrite_chars:",overwrite_chars,"\n")
            #print("transcript + overwrite_chars:",transcript + overwrite_chars,"\n")
           # print(transcript + overwrite_chars)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            

            num_chars_printed = 0
            
            

class InitForm(Form):
    my_txtarea = TextAreaField(default='Begin \n\n\n\n\n\n\n\n\n\n\nEnd',render_kw={'class': 'inline-txtarea1 lined'})
    check_txtarea = TextAreaField(default='OK \n\n\n\n\n\n\n\n\n\n\nOK',render_kw={'class': 'inline-txtarea2 lined'})
    IDNome = StringField('')

    
class EditTextArea(Form):

    my_txtarea = TextAreaField('my_txtarea',render_kw={'class': 'inline-txtarea1 lined'})
    check_txtarea = TextAreaField(render_kw={'class': 'inline-txtarea2 lined'})
    IDNome = StringField('IDNome')

@app.route('/')
def index():
    form = InitForm()
    
    return render_template("index.html", form=form)
    
 
@app.route('/', methods=['POST'])
def submit():
    global text_textarea
    global check_text

    
    check_text = ""

    text_textarea = request.form['my_txtarea']
    #name_file = request.form['nomePrograma']
    name_file = request.form['IDNome']
    text_code = text_textarea    
    
    cod_list = list()
    for var_i in text_code:
        cod_list.append(var_i)
    
    cod_list = [i for i in cod_list if i != '\r']
    text_area_att = ''.join(cod_list)
        
    
    if 'executar' in request.form:
        check_text = code_verifier(text_textarea)
        if ("ERROR" in check_text):
            pass
        else:
            cozmo.run_program(cozmo_program)
              
    if 'BotaoSalvar' in request.form:
        
        if name_file == "":
            pass
            
        elif name_file.isspace() == True:
            pass
            
        else:
            with open(os.path.join(dir_files,name_file + ".txt"),"w") as save_file:
                save_file.write(text_area_att)
        
    else:
        pass
    
        
    #text_code = text_code[0:text_code.find("End")+3]
    form = EditTextArea()
    form.my_txtarea.data = text_area_att
    form.check_txtarea.data = check_text
    form.IDNome.data = name_file
    
    return render_template("index.html", form=form)


def make_text_image(text_to_draw):
    
    font_size = int(10)
    var_loop = True
    
    
    while(var_loop == True):
        width_text,height_text = GetTextDimensions(text_to_draw, font_size, "Arial")

        if (width_text > 110) or (height_text > 28):
            font_size = font_size - 1
            var_loop = False
        else:
            font_size = font_size + 1
    
    
    #font and font size
    _text_font = None
    try:
        _text_font = ImageFont.truetype("arial.ttf",font_size)
    except IOError:
        try:
            _text_font = ImageFont.truetype("/Library/Fonts/Arial.ttf", font_size)
        except IOError:
            pass

    x = 1
    y = 1
    font=_text_font
    '''Make a PIL.Image with the given text printed on it

    Args:
        text_to_draw (string): the text to draw to the image
        x (int): x pixel location
        y (int): y pixel location
        font (PIL.ImageFont): the font to use

    Returns:
        :class:(`PIL.Image.Image`): a PIL image with the text drawn on it
    '''

    # make a blank image for the text, initialized to opaque black
    text_image = Image.new('RGBA', cozmo.oled_face.dimensions(), (0, 0, 0, 255))

    # get a drawing context
    dc = ImageDraw.Draw(text_image)

    # draw the text
    dc.text((x, y), text_to_draw, fill=(255, 255, 255, 255), font=font)

    return text_image

def main():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        return_text = listen_print_loop(responses)
        return return_text
    
def extract_number_word(palavra):
    num_str = ""
    num_list = list()
    
    loop_i = False
    i = 0

    while(loop_i == False):
        
        if (palavra[i].isdigit() == True):
            num_str = num_str + palavra[i]
            if (i == (len(palavra)-1)):
                num_list.append(num_str)
                
        elif (palavra[i]=="."):
            
            if (i == (len(palavra)-1)):
                if (palavra[i-1].isdigit() == True):
                    num_list.append(num_str)
                else:pass
                                    
            elif (i == 0):
                if (palavra[i+1].isdigit() == True):
                    num_str = num_str + palavra[i]
                else:pass
            
            elif (palavra[i-1].isdigit() == True or palavra[i+1].isdigit() == True):
                num_str = num_str + palavra[i]
                    
        elif (palavra[i]== "-"):
            
            if (i == (len(palavra)-1)):pass
        
            elif (palavra[i+1].isdigit() == True):
                num_str = num_str + palavra[i]
                                        
            elif (i < (len(palavra)-2)):
                if (palavra[i+1]==" " and palavra[i+2].isdigit() == True):
                    num_str = num_str + palavra[i] + palavra[i+2] 
                    i = i+2
         
        else:
            if (num_str != ""):
                num_list.append(num_str)
                num_str = ""
        i = i+1
        if (i == len(palavra)):
            loop_i = True
   
    num_list_float = [float(i) for i in num_list]
    return num_list_float


def GetTextDimensions(text, points, font):
    class SIZE(ctypes.Structure):
        _fields_ = [("cx", ctypes.c_long), ("cy", ctypes.c_long)]

    hdc = ctypes.windll.user32.GetDC(0)
    hfont = ctypes.windll.gdi32.CreateFontA(points, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, font)
    hfont_old = ctypes.windll.gdi32.SelectObject(hdc, hfont)

    size = SIZE(0, 0)
    ctypes.windll.gdi32.GetTextExtentPoint32A(hdc, text, len(text), ctypes.byref(size))

    ctypes.windll.gdi32.SelectObject(hdc, hfont_old)
    ctypes.windll.gdi32.DeleteObject(hfont)

    return (size.cx, size.cy)


def checkKeys(myData):
    """test for various keyboard inputs"""
    
    #extract the data
    (event, background, drawColor, lineWidth, keepGoing) = myData
    #print myData
    print(event.key)
    if event.key == pygame.K_KP_ENTER or event.key == 13:
        #quit    
        keepGoing = False
    elif event.key == pygame.K_c:
        #clear screen
        background.fill((255, 255, 255))
    elif event.key == pygame.K_s:
        #save picture
        pygame.image.save(background, "painting.bmp")
    elif event.key == pygame.K_l:
        #load picture
        background = pygame.image.load("painting.bmp")
    elif event.key == pygame.K_r:
        #red
        drawColor = (255, 0, 0)
    elif event.key == pygame.K_g:
        #green
        drawColor = (0, 255, 0)
    elif event.key == pygame.K_w:
        #white
        drawColor = (255, 255, 255)
    elif event.key == pygame.K_b:
        #blue
        drawColor = (0, 0, 255)
    elif event.key == pygame.K_k:
        #black
        drawColor = (0, 0, 0)

    #line widths
    elif event.key == pygame.K_1:
        lineWidth = 1
    elif event.key == pygame.K_2:
        lineWidth = 2
    elif event.key == pygame.K_3:
        lineWidth = 3
    elif event.key == pygame.K_4:
        lineWidth = 4
    elif event.key == pygame.K_5:
        lineWidth = 5
    elif event.key == pygame.K_6:
        lineWidth = 6
    elif event.key == pygame.K_7:
        lineWidth = 7
    elif event.key == pygame.K_8:
        lineWidth = 8
    elif event.key == pygame.K_9:
        lineWidth = 9

    #return all values 
    myData = (event, background, drawColor, lineWidth, keepGoing)
    return myData

def showStats(drawColor, lineWidth):
    """ shows the current statistics """
    myFont = pygame.font.SysFont("None", 20)
    stats = "color: %s, width: %d" % (drawColor, lineWidth)
    statSurf = myFont.render(stats, 1, (drawColor))
    return statSurf


def micros():
    "return a timestamp in microseconds (us)"
    tics = ctypes.c_int64()
    freq = ctypes.c_int64()

    #get ticks on the internal ~2MHz QPC clock
    ctypes.windll.Kernel32.QueryPerformanceCounter(ctypes.byref(tics)) 
    #get the actual freq. of the internal ~2MHz QPC clock
    ctypes.windll.Kernel32.QueryPerformanceFrequency(ctypes.byref(freq))  

    t_us = tics.value*1e6/freq.value
    return t_us

def traj_tracking_controller(dxd, dyd, xd, yd, x, y, phi, a, kx, ky):
    '''TRAJ_TRACKING_CONTROLLER generates commands of linear and angular speeds
    based on the desired speed and position error.
    
       Inputs:
       dxd: x component of desired linear speed [m/s]
       dyd: y component of desired linear speed [m/s]
       xd: x coordinate of desired position [m]
       yd: y coordinate of desired position [m]
       x: x coordinate of actual robot position [m]
       y: y coordinate of actual robot position [m]
       phi: actual robot orientation [rad]
       a: distance of point of interest to the center of the robot [m]
       kx: controller gain for x coordinate
       ky: controller gain for y coordinate'''

    #Controller matrix:
    lin_mat,col_mat= 2,2
    C = [[0 for x in range(col_mat)] for y in range(lin_mat)]
    
    C[0][0] = math.cos(phi)
    C[0][1] = math.sin(phi)
    C[1][0] = -1/a*math.sin(phi)
    C[1][1] = 1/a*math.cos(phi)

    
    #Position error:
    x_err = xd - x
    y_err = yd - y
    

    #If error is smaller than 2 cm, make it null:
    if (abs(x_err) < 0.02) and (abs(y_err) < 0.02):
        x_err = 0
        y_err = 0

    

    
    lin_mat,col_mat= 2,1
    mat_contr = [[0 for x in range(col_mat)] for y in range(lin_mat)]
    
    mat_contr[0][0] = dxd + kx*x_err
    mat_contr[1][0] = dyd + ky*y_err
    
    
    #Controller equation:
    v_ref = multiply_matrix(C,mat_contr,2,1,2)

                 
    return v_ref

def multiply_matrix(A,B,A_i,B_j,B_i):
    

    lin_mat,col_mat= A_i,B_j
    C = [[0 for x in range(col_mat)] for y in range(lin_mat)]
    for i in range (0,A_i):
        for j in range(0,B_j):
            C_aux = 0
            for j_aux in range (0,B_i):
                C_aux = C_aux + A[i][j_aux]*B[j_aux][j]
            C[i][j] = C_aux
            
    return (C)


def code_verifier(txt_code):
    
    
    text_split = txt_code.splitlines()
    tam_list = len(text_split)
    
    text_split = [x_upper.upper() for x_upper in text_split]
    check_code = ""   
    end_if = False
    wait_ans = False
    end_ask = False
    end_if_not = False
    ans_if = False
    
    for i in range (0,tam_list):
        
        if ("BEGIN" in text_split[i]):
            check_code = check_code + "OK\n"
        elif ("END" in text_split[i] and len(text_split[i].replace(" ","")) == 3):
            check_code = check_code + "OK\n"
        
        elif("ASK" in text_split[i]):

            txt_ask = text_split[i].replace(" ","")
            if ("ASK(" in txt_ask) and (txt_ask[0:3] == "ASK") and (txt_ask[len(txt_ask)-1] == ")"):
                
                ask_index = i+1
                while (ask_index < tam_list):
                    if ("WAIT FOR ANSWER" in text_split[ask_index]):
                        wait_ans = True
                    elif ("END ASK" in text_split[ask_index]):
                        end_ask = True
                        
                    elif (wait_ans == True and end_ask == True):
                        ask_index = tam_list
                        check_code = check_code + "OK\n"
                        
                    ask_index = ask_index + 1
                if (wait_ans == False or end_ask == False):
                    check_code = check_code + "ERROR\n"
                    
                    
                    
        elif ("WAIT FOR ANSWER" in text_split[i] and wait_ans == True):
            check_code = check_code + "OK\n"
            
        elif ("END ASK" in text_split[i] and wait_ans == True and end_ask == True):
            check_code = check_code + "OK\n"
            
        elif ("IF ANSWER" in text_split[i] and wait_ans == True and end_ask == True):
            
            if_answer = text_split[i].replace(" ","")
            if ("IFANSWER=" in if_answer):
                ans_if = True
                if(if_answer.find("IFANSWER=") == 0 and len(if_answer) > 9):
                    if_index = i + 1
                    
                    while (if_index < tam_list):
                        if ("END IF" in text_split[if_index] and "NOT" not in text_split[if_index]):
                            end_if = True
                            check_code = check_code + "OK\n"
                            if_index = tam_list
 
                        if_index = if_index + 1
                    if (end_if == False):
                        check_code = check_code + "ERROR\n"
                        
                        

        elif ("IF NOT" in text_split[i] and ans_if == True and "END" not in text_split[i]):

            if_index = i + 1
            while (if_index < tam_list):
                if ("END IF NOT" in text_split[if_index]):
                    end_if_not = True
                    check_code = check_code + "OK\n"
                    if_index = tam_list
 
                if_index = if_index + 1
            if (end_if_not == False):
                check_code = check_code + "ERROR\n"
                
        
        elif ("END IF" in text_split[i] and end_if == True and "NOT" not in text_split[i]):
            check_code = check_code + "OK\n"
            
        elif ("END IF NOT" in text_split[i] and end_if_not == True):
            check_code = check_code + "OK\n"
            
        elif ("DRAW ON SCREEN" in text_split[i]):
            check_code = check_code + "OK\n"
            
        elif ("FIND FACE" in text_split[i]):
            check_code = check_code + "OK\n"
            
        elif ("READ TEXT" in text_split[i]):
            check_code = check_code + "OK\n"
        
        elif "DRAW" in text_split[i]:
            if ("CIRCLE" in text_split[i]) or ("SQUARE" in text_split[i]) or ("RECTANGLE" in text_split[i]) or ("TRIANGLE" in text_split[i]):
                check_code = check_code + "OK\n"
            else:
                check_code = check_code + "ERROR\n"
                
        elif "GAME" in text_split[i]:
            if ("CIRCLE" in text_split[i]) or ("SQUARE" in text_split[i]) or ("RECTANGLE" in text_split[i]) or ("TRIANGLE" in text_split[i]):
                check_code = check_code + "OK\n"
            else:
                check_code = check_code + "ERROR\n"
                
        elif ("SPEAK" in text_split[i]):
            txt_speak = text_split[i].replace(" ","")
            if ("SPEAK(" in txt_speak) and (txt_speak[0:3] == "SPE") and (txt_speak[len(txt_speak)-1] == ")"):
                check_code = check_code + "OK\n"
                
        elif ("ANIMATION" in text_split[i]):
            txt_anim = text_split[i].replace(" ","")
            if ("ANIMATION(" in txt_anim) and (txt_anim[0:3] == "PLA") and (txt_anim[len(txt_anim)-1] == ")"):
                check_code = check_code + "OK\n"
                
        elif ("VIDEO" in text_split[i]):
            txt_video = text_split[i].replace(" ","")
            if ("VIDEO(" in txt_video) and (txt_video[0:3] == "VID") and (txt_video[len(txt_video)-1] == ")"):
                check_code = check_code + "OK\n"
                
        elif ("FIND" in text_split[i]):
            txt_find = text_split[i].replace(" ","")
            if ("FIND(" in txt_find) and (txt_find[0:3] == "FIN") and (txt_find[len(txt_find)-1] == ")"):
                check_code = check_code + "OK\n"
                
        
        elif("FORWARD" in text_split[i]) or ("BACKWARD" in text_split[i]) or ("LEFT" in text_split[i]) or ("RIGHT" in text_split[i]):
            move = text_split[i].replace(" ","")
            move_dist = move[move.find("(")+1:move.find(")")]
            try:
                float(move_dist)
                numb_float = True
            except ValueError:
                numb_float = False

            if (numb_float == True):
                check_code = check_code + "OK\n"
                
            else:
                check_code = check_code + "ERROR\n"
                
        elif("PEN UP" in text_split[i]) or ("PEN DOWN" in text_split[i]):
            check_code = check_code + "OK\n"
            
        elif("HEAD UP" in text_split[i]) or ("HEAD DOWN" in text_split[i]):
            check_code = check_code + "OK\n"
                
        elif("SUM" in text_split[i]):
            sum_num = text_split[i].replace(" ","")
            if ("SUM(" in sum_num) and ("+" in sum_num) and (")" in sum_num) and (sum_num[0:3] == "SUM") and (sum_num[len(sum_num)-1] == ")"):
                x = sum_num[sum_num.find("SUM(")+4:sum_num.find("+")]
                y = sum_num[sum_num.find("+")+1:sum_num.find(")")]
                print(x)
                print(y)
                try:
                    float(x)
                    float(y)
                    numb_float = True
                except ValueError:
                    numb_float = False
                    
                    
                if (numb_float == True):
                    check_code = check_code + "OK\n"
                else:
                    check_code = check_code + "ERROR\n"
            else:
                check_code = check_code + "ERROR\n"
                
                
        elif("SUB" in text_split[i]):
            sum_num = text_split[i].replace(" ","")
            if ("SUB(" in sum_num) and ("-" in sum_num) and (")" in sum_num) and (sum_num[0:3] == "SUB") and (sum_num[len(sum_num)-1] == ")"):
                x = sum_num[sum_num.find("SUB(")+4:sum_num.find("-")]
                y = sum_num[sum_num.find("-")+1:sum_num.find(")")]
                print(x)
                print(y)
                try:
                    float(x)
                    float(y)
                    numb_float = True
                except ValueError:
                    numb_float = False
                    
                    
                if (numb_float == True):
                    check_code = check_code + "OK\n"
                else:
                    check_code = check_code + "ERROR\n"
            else:
                check_code = check_code + "ERROR\n"
                
                
        elif("MULT" in text_split[i]):
            sum_num = text_split[i].replace(" ","")
            if ("MULT(" in sum_num) and ("*" in sum_num) and (")" in sum_num) and (sum_num[0:4] == "MULT") and (sum_num[len(sum_num)-1] == ")"):
                x = sum_num[sum_num.find("MULT(")+5:sum_num.find("*")]
                y = sum_num[sum_num.find("*")+1:sum_num.find(")")]
                print(x)
                print(y)
                try:
                    float(x)
                    float(y)
                    numb_float = True
                except ValueError:
                    numb_float = False
                    
                    
                if (numb_float == True):
                    check_code = check_code + "OK\n"
                else:
                    check_code = check_code + "ERROR\n"
            else:
                check_code = check_code + "ERROR\n"
                
                
                
                
        elif("DIV" in text_split[i]):
            sum_num = text_split[i].replace(" ","")
            if ("DIV(" in sum_num) and ("/" in sum_num) and (")" in sum_num) and (sum_num[0:3] == "DIV") and (sum_num[len(sum_num)-1] == ")"):
                x = sum_num[sum_num.find("DIV(")+4:sum_num.find("/")]
                y = sum_num[sum_num.find("/")+1:sum_num.find(")")]
                print(x)
                print(y)
                try:
                    float(x)
                    float(y)
                    numb_float = True
                except ValueError:
                    numb_float = False
                    
                    
                if (numb_float == True):
                    check_code = check_code + "OK\n"
                else:
                    check_code = check_code + "ERROR\n"
            else:
                check_code = check_code + "ERROR\n"

        elif ("QUADRATIC EQUATION" in text_split[i]):
            quadratic_eq = text_split[i].replace(" ","")
            if ("(A=" in quadratic_eq) and (",B=" in quadratic_eq) and (",C=" in quadratic_eq) and ("(" in quadratic_eq) and (")" in quadratic_eq) and (quadratic_eq[0:3] == "QUA") and (quadratic_eq[len(quadratic_eq)-1] == ")"):
                a = quadratic_eq[quadratic_eq.find("(A=")+3:quadratic_eq.find(",B=")]
                b = quadratic_eq[quadratic_eq.find(",B=")+3:quadratic_eq.find(",C=")]
                c = quadratic_eq[quadratic_eq.find(",C=")+3:quadratic_eq.find(")")] 
                
                try:
                    float(a)
                    float(b)
                    float(c)
                    numb_float = True
                except ValueError:
                    numb_float = False     
                    
                if (numb_float == True):
                    check_code = check_code + "OK\n"
                else:
                    check_code = check_code + "ERROR\n"
            else:
                check_code = check_code + "ERROR\n"
                

        elif ("SPELL WORD" in text_split[i]):
            txt_speak = text_split[i].replace(" ","")
            if ("WORD(" in txt_speak) and (txt_speak[0:3] == "SPE") and (txt_speak[len(txt_speak)-1] == ")"):
                check_code = check_code + "OK\n"
        
        elif (text_split[i] == ""):
            check_code = check_code + "\n"
        else:
            check_code = check_code + "ERROR\n"
            
        
    return check_code
            
    
def get_in_position(robot: cozmo.robot.Robot):
    '''If necessary, Move Cozmo's Head and Lift to make it easy to see Cozmo's face.'''
    if (robot.lift_height.distance_mm > 45) or (robot.head_angle.degrees < 40):
        with robot.perform_off_charger():
            lift_action = robot.set_lift_height(0.0, in_parallel=True)
            head_action = robot.set_head_angle(cozmo.robot.MAX_HEAD_ANGLE,
                                               in_parallel=True)
            lift_action.wait_for_completed()
            head_action.wait_for_completed()
            
            
def cozmo_program(robot: cozmo.robot.Robot):
    

    global text_split
    global move_speed_int
    global eq_solved
    
    text_split = text_textarea.splitlines()
    text_split_web = text_split
    tam_list = len(text_split)
    text_split = [x_upper.upper() for x_upper in text_split]

    
    var_if = False
    i = -1
    script_results = ""


    #for i in range (0,tam_list):
    while (i<(tam_list-1)):
        i = i+1
        
        script_results = script_results + text_split[i] +"\n"
        
                    
        if ("WAIT FOR ANSWER" in text_split[i]):
            text_ans_ask =  main()
            script_results = script_results + "Answer given: " + text_ans_ask +"\n"
            text_ans_ask = text_ans_ask.upper()
            text_ans_ask = text_ans_ask.replace(" ","")
            
        elif ("IF ANSWER" in text_split[i]):
            correct_ans = text_split[i].replace(" ","")   
            correct_ans = correct_ans[correct_ans.find("=")+1:len(correct_ans)]

            
            if (correct_ans.isdigit() == True):
                if (text_ans_ask == correct_ans):
                    var_if = True
                else:
                    while (True):
                        if "END IF" in text_split[i]:
                            break
                        else:
                            if (i<(tam_list-1)):
                                
                                i = i+1
                            else:
                                break
            else:
                if (text_ans_ask in correct_ans):
                    var_if = True
                else:
                    while (True):
                        if "END IF" in text_split[i]:
                            break
                        else:
                            if (i<(tam_list-1)):
                                
                                i = i+1
                            else:
                                break
                            
        elif ("IF NOT" in text_split[i] and var_if == True):

            while (True):
                if "END IF NOT" in text_split[i]:
                    break
                else:
                    if (i<(tam_list-1)):
                        
                        i = i+1
                    else:
                        break

                    
        
        elif("SCREEN" in text_split[i]):

            
            pygame.init()
            screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("Paint:  (r)ed, (g)reen, (b)lue, (w)hite, blac(k), (1-9) width, (c)lear, (s)ave, (l)oad, (q)uit")
            
            background = pygame.Surface(screen.get_size())
            background.fill((255, 255, 255))
            
            clock = pygame.time.Clock()
            
            lineStart = (0, 0)
            drawColor = (0, 0, 0)
            lineWidth = 3
            
            list_pos_x = list()
            list_pos_y = list()
            
            pos_init = True
            
            
            close_window = False
            draw_mode = True
            
            while (draw_mode == True):
                
                keepGoing = True
            
                while keepGoing:
                    clock.tick(30)
                    
                    for event in pygame.event.get():
                        
                        if event.type == pygame.QUIT:
                            keepGoing = False
                            close_window = True
                            draw_mode = False
                            
                            
                            
                        elif event.type == pygame.MOUSEMOTION:
                            lineEnd = pygame.mouse.get_pos()
                            if pygame.mouse.get_pressed() == (1, 0, 0):
                                pygame.draw.line(background, drawColor, lineStart, lineEnd, lineWidth)
                                if (pos_init == True):
                                    var_x_init = lineStart[0]
                                    var_y_init = lineStart[1]
                                    pos_init = False
                                var_x = lineStart[0]
                                var_y = lineStart[1]
                                list_pos_x.append(var_x-var_x_init)
                                list_pos_y.append(-(var_y - var_y_init))
                                    
                            
                            lineStart = lineEnd
                        elif event.type == pygame.KEYDOWN:
                            myData = (event, background, drawColor, lineWidth, keepGoing)
                            myData = checkKeys(myData)
                            (event, background, drawColor, lineWidth, keepGoing) = myData
                            if (event.key == pygame.K_c):
                                list_pos_x = list()
                                list_pos_y = list()
                                pos_init = True
                            elif event.key == pygame.K_q:
                                keepGoing = False
                                close_window = True
                                draw_mode = False
                                

                    screen.blit(background, (0, 0))
                    myLabel = showStats(drawColor, lineWidth)
                    screen.blit(myLabel, (450, 450))
                    pygame.display.flip()
                if close_window == True:
                    pygame.quit()    
                #return (list_pos_x,list_pos_y,myData)
                
                #var_x,var_y,myData= DrawMode(data_draw,init_draw)
                var_x = list_pos_x
                var_y = list_pos_y
                
                #a = 30
                a = 40
                pose = robot.pose
                phi_init = pose.rotation.angle_z.radians
                x_init,y_init,z_init = pose.position.x_y_z
                x_init = x_init + a*math.cos(phi_init)
                y_init = y_init + a*math.sin(phi_init)
                           
                x = 0
                y = 0
                phi = 0
                
                dxd = 0
                dyd = 0
                     
                #Initial robot speeds:
                dx = 0
                dy = 0
                
                #Initial robot pose ant
                x_ant = 0
                y_ant = 0
                phi_ant = 0
            
                
                #Controller gains:
                kx = 0.6
                ky = 0.6
                
                #Vectors to store simulation data:
                
                x_vector = list()
                y_vector = list()
                phi_vector = list()
                u_vector = list()
                w_vector = list()
                u_ref_vector = list()
                w_ref_vector = list()
                dx_vector = list()
                dy_vector = list()
                
                xd_vector = list()
                yd_vector = list()
                
                t_vector = list()
                
                tempo_perc = 0
                
                
                #Simulation loop:
                #for t in np.arange(0.0,t_final, ts):
                time_0 = time.time()
                init_var = True
                
                var_dist = False
                cal_a_b = True
                
                var_x_final = []
                var_y_final = []
                ind_x_y = 0
                
                t_delay = 0.03
                
                
            
                #for ind_x_y in range (0,len(var_x)):
                while (ind_x_y<=len(var_x)-1 and close_window == False):
                    if (ind_x_y<len(var_x)-1):
                        if (math.sqrt(pow(var_x[ind_x_y + 1] - var_x[ind_x_y] ,2) + pow(var_y[ind_x_y + 1] - var_y[ind_x_y],2)) > 30):
                            var_x_final =  var_x_final + np.linspace(var_x[ind_x_y-1],var_x[ind_x_y],50).tolist()
                            var_y_final =  var_y_final + np.linspace(var_y[ind_x_y-1],var_y[ind_x_y],50).tolist()
                            pass
    
    
                        else:
                            var_x_final =  var_x_final + np.linspace(var_x[ind_x_y],var_x[ind_x_y+1],6).tolist()
                            var_y_final =  var_y_final + np.linspace(var_y[ind_x_y],var_y[ind_x_y+1],6).tolist()
                        
                    else:
                        
                        var_x_final =  var_x_final + np.linspace(var_x[ind_x_y-1],var_x[ind_x_y],50).tolist()
                        var_y_final =  var_y_final + np.linspace(var_y[ind_x_y-1],var_y[ind_x_y],50).tolist()
                        
                    ind_x_y = ind_x_y + 1
                    
    
                        
                if (close_window == False):         
                    robot.set_lift_height(0).wait_for_completed()
                ind_x_y = 0
                
                
                
                while (ind_x_y<=len(var_x_final)-2 and close_window == False):
                    
                    
                    time.sleep(t_delay)                
                            
                    if (var_dist == False):
                    #Get the desired position and speed for time t:
    
                        xd,yd = var_x_final[ind_x_y],var_y_final[ind_x_y]
                    
    
                    
                    dxd = (var_x_final[i+1] - var_x_final[i])/t_delay
                    dyd = (var_y_final[i+1] - var_y_final[i])/t_delay
                    #Call the controller to generate reference commands u_ref and w_ref:
                    v_ref = traj_tracking_controller(dxd, dyd, xd, yd, x, y, phi, a, kx, ky)
                    
                    
                    #Apply this signal in the robot
                    u_ref = v_ref[0][0]
                    w_ref = v_ref[1][0]
                    
                    #Calculate robot speeds on absolute reference frame using robot model:
                    #speeds = update_robot_speeds(u_ref, w_ref, phi, a)
                    
                    
                    
                    '''UPDATE_ROBOT_SPEEDS updates the speed of the differential drive mobile
                    robot according to its kinematic model.
                       Inputs:
                       u: robot linear velocity [m/s]
                       w: robot angular velocity [rad/s]
                       phi: robot orientation [rad]
                       a: distance of point of interest to the center of the robot [m]
                      
                       Outputs:
                       speeds: vector containing   x component of robot linear speed [m/s]
                                                   y component of robot linear speed [m/s]
                                                   angular speed [rad/s]'''
                
                    u_max = 220    # max linear speed [mm/s]    
                    w_max = 9.82 # max angular speed [rad/s]
                    # If speed commands from the controller are higher than the maximum speeds
                    # achivable by the robot, saturate:
                    u = np.sign(u_ref)*min(abs(u_ref), u_max)
                    w = np.sign(w_ref)*min(abs(w_ref), w_max)
                    
                
                    v_right = u + (44.8/2)*w
                    v_left = u - (44.8/2)*w
                    
                    '''if (v_right<0 and v_right >-10):
                        v_right = -10
                        
                    if (v_right>0 and v_right < 10):
                        v_right = 10
                        
                    if (v_left<0 and v_left > -10):
                        v_left = -10
                        
                    if (v_left>0 and v_left < 10):
                        v_left = 10'''
                        
                    robot.drive_wheels(v_left,v_right)
                    #robot.drive_wheels(v_left,v_right)
                    
                    pose = robot.pose
                    phi_now = pose.rotation.angle_z.radians
                    x_now,y_now,z_now = pose.position.x_y_z
                    x_now = x_now + a*math.cos(phi_now)
                    y_now = y_now + a*math.sin(phi_now)
                    
                    
                    x = x_now - x_init
                    y = y_now - y_init
            
                    phi = phi_now - phi_init
                    
                    x_final = x - x_ant
                    y_final = y - y_ant
                      
                    if (init_var == True):
                        time_ant = 0
                        init_var = False
                        
                        
                    time_now = micros()/1000000
                    
                    tot_time = time_now - time_ant
                    time_ant = time_now
                    
                    dx = x_final/tot_time
                    dy = y_final/tot_time
                    
                    
                    v_right_real = robot.right_wheel_speed.speed_mmps
                    v_left_real = robot.left_wheel_speed.speed_mmps
                    u = (v_right_real + v_left_real)/2
                    u = float(u)
                    #u = math.sqrt(math.pow(abs(dx),2) +  math.pow(abs(dy),2))
                    w = (phi-phi_ant)/tot_time
                    
                    
                    x_ant = x
                    y_ant = y
                    phi_ant = phi
    
                    #Keep phi within [-pi,pi]:
                    if (phi > math.pi):
                        phi = phi - 2*math.pi
                    else:
                        if (phi < -math.pi):
                            phi = phi + 2*math.pi
                            
                            
                    
                    if (math.sqrt(pow(var_x_final[ind_x_y + 1] - var_x_final[ind_x_y] ,2) + pow(var_y_final[ind_x_y + 1] - var_y_final[ind_x_y],2)) > 20):                    
                        if (math.sqrt(pow(var_x_final[ind_x_y + 1] - x ,2) + pow(var_y_final[ind_x_y + 1] - y,2)) > 20):
                            
                            
                            if (cal_a_b == True):
    
                                robot.set_lift_height(0.2).wait_for_completed()
                                cal_a_b = False
                                dist_x_y = math.sqrt(pow(var_x_final[ind_x_y + 1] - var_x_final[ind_x_y] ,2) + pow(var_y_final[ind_x_y + 1] - var_y_final[ind_x_y],2))
                                dist_x_y = int(dist_x_y)
                                
                                var_x_no_draw =  np.linspace(var_x_final[ind_x_y],var_x_final[ind_x_y + 1],dist_x_y).tolist()
                                var_y_no_draw =  np.linspace(var_y_final[ind_x_y],var_y_final[ind_x_y + 1],dist_x_y).tolist()
                                ind_no_draw = 0
                                #print(var_x_no_draw)
                                time.sleep(0.5)
                            
                            
                            
                            if (ind_no_draw<len(var_x_no_draw) - 2):
                                xd = var_x_no_draw[ind_no_draw]
                                yd = var_y_no_draw[ind_no_draw]
                                
                                dxd = 0
                                dyd = 0
                                var_dist = True
                                
                            ind_no_draw = ind_no_draw + 1
                                
                        else:
                            #time.sleep(0.5)
                            robot.set_lift_height(0).wait_for_completed()
                            var_dist = False
                            cal_a_b = True
                            
    
                    #Store simulation data:
            
                    if (var_dist == False):
                        x_vector.append(x)
                        y_vector.append(y)
                        phi_vector.append(phi)
                        u_vector.append(u)
                        w_vector.append(w)
                        u_ref_vector.append(u_ref)
                        w_ref_vector.append(w_ref)
                        dx_vector.append(dx)
                        dy_vector.append(dy)
                        
                        xd_vector.append(xd)
                        yd_vector.append(yd)
                        
                        tempo_perc = time.time() - time_0
                        t_vector.append(tempo_perc)
                        
                        
                        ind_x_y = ind_x_y + 1
         
                #t_vector = np.arange(0.0,t_final, ts)
                #t_vector = [0:ts:t_final]'
                    
                #Plot results:
                
                if (close_window == False):
                    plt.figure(1)
                    plt.plot(t_vector,u_vector)
                    plt.title('Linear speed')
                    plt.ylabel('u [m/s]')
                    plt.xlabel('time [s]')
                    plt.grid(True)
                    
                    plt.figure(2)
                    plt.plot(t_vector,w_vector)
                    plt.title('Angular speed')
                    plt.ylabel('ω [rad/s]')
                    plt.xlabel('time [s]')
                    plt.grid(True)
                    
                    plt.figure(3)
                    plt.plot(x_vector, y_vector)
                    plt.title('Robot path')
                    plt.xlabel('x [mm]')
                    plt.ylabel('y [mm]')
                    plt.grid(True)
                    
                    plt.figure(4)
                    plt.plot(xd_vector, yd_vector)
                    plt.title('Robot path')
                    plt.xlabel('x [mm]')
                    plt.ylabel('y [mm]')
                    plt.grid(True)
                    
                    plt.show()
                    
                    robot.set_lift_height(0.2).wait_for_completed()
                    robot.abort_all_actions()
                    robot.drive_wheels(0,0)
                
                
            
                
            

        elif ("PLAY VIDEO" in text_split[i]):
            play_video = text_split_web[i]
            play_video = play_video[play_video.find("(")+1:play_video.find(")")]
            webbrowser.open(play_video, new=2)
            
        elif("PLAY ANIMATION" in text_split[i]):
            play_animation = text_split[i].replace(" ","")
            play_animation = play_animation[play_animation.find("(")+1:play_animation.find(")")]
            
            if (play_animation == "HAPPY"):
                robot.play_anim(name="anim_meetcozmo_celebration_02").wait_for_completed()
            elif (play_animation == "SAD"):
                robot.play_anim(name="anim_memorymatch_failgame_cozmo_03").wait_for_completed()
            elif (play_animation == "ANGRY"):
                robot.play_anim(name="anim_keepaway_losegame_03").wait_for_completed()
            elif (play_animation == "SURPRISE"):
                robot.play_anim(name="anim_reacttppl_surprise").wait_for_completed()
            elif (play_animation == "BORED"):
                robot.play_anim(name="anim_bored_event_01").wait_for_completed()
            elif (play_animation == "SLEEP"):
                robot.play_anim(name="anim_gotosleep_getin_01").wait_for_completed()
                robot.play_anim(name="anim_gotosleep_sleeping_01").wait_for_completed()
            elif (play_animation == "WAKE"):
                robot.play_anim(name="anim_gotosleep_getout_02").wait_for_completed()
                
                
                
        elif "GAME" in text_split[i]:
            robot.set_lift_height(0).wait_for_completed()
            

            if "CIRCLE" in text_split[i]:
            
                robot.drive_wheels(150,0)
                time.sleep(4.7)
                robot.drive_wheels(0,0)
                robot.set_lift_height(0.2).wait_for_completed()
                
                robot.say_text("What is the name of this geometric shape?").wait_for_completed()
                text_recog =  main()
                
                script_results = script_results + "Question asked: What is the name of this geometric shape?" + "\n"
                script_results = script_results + "Answer given: " + text_recog + "\n"
                             
                text_recog =  text_recog.upper()
                if ("CIRCLE" in text_recog):
                    robot.play_anim(name="anim_rtpmemorymatch_yes_03").wait_for_completed()
                    robot.say_text("Is a circle's shape round?").wait_for_completed()
                    text_recog =  main()
                    
                    script_results = script_results + "Question asked: Is a circle's shape round?" + "\n"
                    script_results = script_results + "Answer given: " + text_recog + "\n"
                
                    text_recog =  text_recog.upper()
                    if ("YES" in text_recog):
                        robot.play_anim(name="anim_rtpmemorymatch_yes_02").wait_for_completed()
                        robot.set_lift_height(0.2).wait_for_completed()
                        
                        robot.say_text("Does a circle have straight lines?").wait_for_completed()
                        text_recog =  main()
                        
                        script_results = script_results + "Question asked: Does a circle have straight lines?" + "\n"
                        script_results = script_results + "Answer given: " + text_recog + "\n"
                    
                        text_recog =  text_recog.upper()
                        if ("NO" in text_recog):
                            robot.play_anim(name="anim_memorymatch_solo_successgame_player_01").wait_for_completed()
                            robot.say_text("Congratulations. You know a lot of things about circles!").wait_for_completed()
                        else:              
                            robot.play_anim(name="anim_rtpmemorymatch_no_01").wait_for_completed()
                            robot.say_text("Too bad... That is not the right answer. The right answer is no. Watch the video on the computer to understand more about circles!").wait_for_completed()
                           
                            webbrowser.open('https://www.youtube.com/watch?v=BmtU1SObpKI', new=2)
                    else:
                        robot.play_anim(name="anim_rtpmemorymatch_no_01").wait_for_completed()
                        robot.say_text("Too bad... That is not the right answer. The right answer is yes. Watch the video on the computer to understand more about circles!").wait_for_completed()
                       
                        webbrowser.open('https://www.youtube.com/watch?v=BmtU1SObpKI', new=2)
      
                else:
                    robot.play_anim(name="anim_rtpmemorymatch_no_01").wait_for_completed()
                    robot.say_text("Too bad... That is not the right answer. The name of this geometric shape is circle. Watch the video on the computer to understand more about circles!").wait_for_completed()
                   
                    webbrowser.open('https://www.youtube.com/watch?v=BmtU1SObpKI', new=2)
            
                
     
            elif "SQUARE" in text_split[i]:
                
                for drive_x in range(4):
                    robot.drive_straight(distance_mm(108), speed_mmps(100)).wait_for_completed()
                    robot.set_lift_height(0.2).wait_for_completed()
                    robot.drive_straight(distance_mm(42), speed_mmps(100)).wait_for_completed()
                    robot.turn_in_place(degrees(-90)).wait_for_completed()
                    robot.drive_straight(distance_mm(-41), speed_mmps(100)).wait_for_completed()
                    robot.set_lift_height(0).wait_for_completed()
                robot.set_lift_height(0.2).wait_for_completed()
                
                robot.say_text("What is the name of this geometric shape?").wait_for_completed()
                text_recog =  main()
                
                script_results = script_results + "Question asked: What is the name of this geometric shape?" + "\n"
                script_results = script_results + "Answer given: " + text_recog + "\n"
                        
                text_recog =  text_recog.upper()
                
                if ("SQUARE" in text_recog):
                    robot.play_anim(name="anim_rtpmemorymatch_yes_03").wait_for_completed()
                    robot.say_text("How many lines does a square shape have?").wait_for_completed()
                    text_recog =  main()
                    
                    script_results = script_results + "Question asked: How many lines does a square shape have?" + "\n"
                    script_results = script_results + "Answer given: " + text_recog + "\n"
                
                    text_recog =  text_recog.upper()
                    if ("4" in text_recog) or ("FOUR" in text_recog):
                        robot.play_anim(name="anim_rtpmemorymatch_yes_02").wait_for_completed()
                        robot.say_text("How many equal lines does a square shape have?").wait_for_completed()
                        text_recog =  main()
                        
                        script_results = script_results + "Question asked: How many equal lines does a square shape have?" + "\n"
                        script_results = script_results + "Answer given: " + text_recog + "\n"
                    
                        text_recog =  text_recog.upper()
                        if ("4" in text_recog) or ("FOUR" in text_recog):
                            robot.play_anim(name="anim_memorymatch_solo_successgame_player_01").wait_for_completed()
                            robot.say_text("Congratulations. You know a lot of things about squares!").wait_for_completed()
                        else:
                            robot.play_anim(name="anim_rtpmemorymatch_no_01").wait_for_completed()
                            robot.say_text("Too bad... That is not the right answer. The right answer is 4. Watch the video on the computer to understand more about squares!").wait_for_completed()
                           
                            webbrowser.open('https://www.youtube.com/watch?v=WHypLi16j4o', new=2)
                    else:
                        robot.play_anim(name="anim_rtpmemorymatch_no_01").wait_for_completed()
                        robot.say_text("Too bad... That is not the right answer. The right answer is 4. Watch the video on the computer to understand more about squares!").wait_for_completed()
                       
                        webbrowser.open('https://www.youtube.com/watch?v=WHypLi16j4o', new=2)
                else:
                    robot.play_anim(name="anim_rtpmemorymatch_no_01").wait_for_completed()
                    robot.say_text("Too bad... That is not the right answer. The name of this geometric shape is square. Watch the video on the computer to understand more about squares!").wait_for_completed()
                   
                    webbrowser.open('https://www.youtube.com/watch?v=WHypLi16j4o', new=2)
      
            elif "TRIANGLE" in text_split[i]:
                for x_drive in range(3):
                    robot.drive_straight(distance_mm(122), speed_mmps(200)).wait_for_completed()
                    robot.set_lift_height(0.2).wait_for_completed()
                    robot.drive_straight(distance_mm(43), speed_mmps(200)).wait_for_completed()
                    robot.turn_in_place(degrees(-120)).wait_for_completed()
                    robot.drive_straight(distance_mm(-40), speed_mmps(200)).wait_for_completed()
                    robot.set_lift_height(0).wait_for_completed()
                robot.set_lift_height(0.2).wait_for_completed()
                    
                robot.say_text("What is the name of this geometric shape?").wait_for_completed()
                text_recog =  main()
                
                script_results = script_results + "Question asked: What is the name of this geometric shape?" + "\n"
                script_results = script_results + "Answer given: " + text_recog + "\n"
                        
                text_recog =  text_recog.upper()
                if ("TRIANGLE" in text_recog):
                    robot.play_anim(name="anim_rtpmemorymatch_yes_03").wait_for_completed()
                    robot.say_text("How many lines does a triangle shape have?").wait_for_completed()
                    text_recog =  main()
                    
                    script_results = script_results + "Question asked: How many lines does a triangle shape have?" + "\n"
                    script_results = script_results + "Answer given: " + text_recog + "\n"
                
                    text_recog =  text_recog.upper()
                    if ("3" in text_recog) or ("THREE" in text_recog):
                        robot.play_anim(name="anim_rtpmemorymatch_yes_02").wait_for_completed()
                        robot.say_text("Is a triangle's shape round?").wait_for_completed()
                        text_recog =  main()
                        
                        script_results = script_results + "Question asked: Is a triangle's shape round?" + "\n"
                        script_results = script_results + "Answer given: " + text_recog + "\n"
                    
                        text_recog =  text_recog.upper()
                        if ("NO" in text_recog):
                            robot.play_anim(name="anim_memorymatch_solo_successgame_player_01").wait_for_completed()
                            robot.say_text("Congratulations. You know a lot of things about triangles!").wait_for_completed()
                        else:
                            robot.play_anim(name="anim_rtpmemorymatch_no_01").wait_for_completed()
                            robot.say_text("Too bad... That is not the right answer. The right answer is no. Watch the video on the computer to understand more about triangles!").wait_for_completed()
                            webbrowser.open('https://www.youtube.com/watch?v=AEpHfWFcfuw', new=2)
                    else:
                        robot.play_anim(name="anim_rtpmemorymatch_no_01").wait_for_completed()
                        robot.say_text("Too bad... That is not the right answer. The right answer is 3. Watch the video on the computer to understand more about triangles!").wait_for_completed()
                        
                        webbrowser.open('https://www.youtube.com/watch?v=AEpHfWFcfuw', new=2)
                else:
                    robot.play_anim(name="anim_rtpmemorymatch_no_01").wait_for_completed()
                    robot.say_text("Too bad... That is not the right answer. The name of this geometric shape is triangle. Watch the video on the computer to understand more about triangles!").wait_for_completed()
                    
                    webbrowser.open('https://www.youtube.com/watch?v=AEpHfWFcfuw', new=2)
                            
            
  
            elif "RECTANGLE" in text_split[i]:
                for drive_x in range(2):
                    robot.drive_straight(distance_mm(135), speed_mmps(100)).wait_for_completed()
                    robot.set_lift_height(0.2).wait_for_completed()
                    robot.drive_straight(distance_mm(45), speed_mmps(100)).wait_for_completed()
                    robot.turn_in_place(degrees(-90)).wait_for_completed()
                    robot.drive_straight(distance_mm(-37), speed_mmps(100)).wait_for_completed()
                    robot.set_lift_height(0).wait_for_completed()
                    
                    robot.drive_straight(distance_mm(45), speed_mmps(100)).wait_for_completed()
                    robot.set_lift_height(0.2).wait_for_completed()
                    robot.drive_straight(distance_mm(45), speed_mmps(100)).wait_for_completed()
                    robot.turn_in_place(degrees(-90)).wait_for_completed()
                    robot.drive_straight(distance_mm(-40), speed_mmps(100)).wait_for_completed()
                    robot.set_lift_height(0).wait_for_completed()
                robot.set_lift_height(0.2).wait_for_completed()
                    
                robot.say_text("What is the name of this geometric shape?").wait_for_completed()
                text_recog =  main()
                
                script_results = script_results + "Question asked: What is the name of this geometric shape?" + "\n"
                script_results = script_results + "Answer given: " + text_recog + "\n"
                    
                text_recog =  text_recog.upper()
                if ("RECTANGLE" in text_recog):
                    robot.play_anim(name="anim_rtpmemorymatch_yes_03").wait_for_completed()
                    robot.say_text("How many lines does a rectangle shape have?").wait_for_completed()
                    text_recog =  main()
                    
                    script_results = script_results + "Question asked: How many lines does a rectangle shape have?" + "\n"
                    script_results = script_results + "Answer given: " + text_recog + "\n"
                    
                    text_recog =  text_recog.upper()
                    if ("4" in text_recog) or ("FOUR" in text_recog):
                        robot.play_anim(name="anim_rtpmemorymatch_yes_02").wait_for_completed()
                        robot.say_text("Is a rectangle shape the same as a square shape?").wait_for_completed()
                        text_recog =  main()
                        
                        script_results = script_results + "Question asked: Is a rectangle shape the same as a square shape?" + "\n"
                        script_results = script_results + "Answer given: " + text_recog + "\n"
                    
                        text_recog =  text_recog.upper()
                        if ("NO" in text_recog):
                            robot.play_anim(name="anim_memorymatch_solo_successgame_player_01").wait_for_completed()
                            robot.say_text("Congratulations. You know a lot of things about rectangles!").wait_for_completed()
                        else:
                            robot.play_anim(name="anim_rtpmemorymatch_no_01").wait_for_completed()
                            robot.say_text("Too bad. That is not the right answer. The square shape has 4 equal lines and the rectangle shape does not have 4 equal lines. Watch the video on the computer to understand more about rectangles!").wait_for_completed()
                           
                            webbrowser.open('https://www.youtube.com/watch?v=cW5muVaoK4I', new=2)
                    else:
                        robot.play_anim(name="anim_rtpmemorymatch_no_01").wait_for_completed()
                        robot.say_text("Too bad. That is not the right answer. The right answer is 4. Watch the video on the computer to understand more about rectangles!").wait_for_completed()
                   
                        webbrowser.open('https://www.youtube.com/watch?v=cW5muVaoK4I', new=2)
                else:
                    robot.play_anim(name="anim_rtpmemorymatch_no_01").wait_for_completed()
                    robot.say_text("Too bad. That is not the right answer. The name of this geometric shape is rectangle. Watch the video on the computer to understand more aboutt rectangles!").wait_for_completed()
                 
                    webbrowser.open('https://www.youtube.com/watch?v=cW5muVaoK4I', new=2)
                    
                    
        elif "DRAW" in text_split[i]:
            robot.set_lift_height(0).wait_for_completed()
        

            if "CIRCLE" in text_split[i]:
            
                robot.drive_wheels(150,0)
                time.sleep(4.7)
                robot.drive_wheels(0,0)
                robot.set_lift_height(0.2).wait_for_completed()
                
            

            elif "SQUARE" in text_split[i]:
                
                for drive_x in range(4):
                    robot.drive_straight(distance_mm(108), speed_mmps(100)).wait_for_completed()
                    robot.set_lift_height(0.2).wait_for_completed()
                    robot.drive_straight(distance_mm(42), speed_mmps(100)).wait_for_completed()
                    robot.turn_in_place(degrees(-90)).wait_for_completed()
                    robot.drive_straight(distance_mm(-41), speed_mmps(100)).wait_for_completed()
                    robot.set_lift_height(0).wait_for_completed()
                robot.set_lift_height(0.2).wait_for_completed()
                
      
            elif "TRIANGLE" in text_split[i]:
                for x_drive in range(3):
                    robot.drive_straight(distance_mm(122), speed_mmps(200)).wait_for_completed()
                    robot.set_lift_height(0.2).wait_for_completed()
                    robot.drive_straight(distance_mm(43), speed_mmps(200)).wait_for_completed()
                    robot.turn_in_place(degrees(-120)).wait_for_completed()
                    robot.drive_straight(distance_mm(-40), speed_mmps(200)).wait_for_completed()
                    robot.set_lift_height(0).wait_for_completed()
                robot.set_lift_height(0.2).wait_for_completed()
                    

  
            elif "RECTANGLE" in text_split[i]:
                for drive_x in range(2):
                    robot.drive_straight(distance_mm(135), speed_mmps(100)).wait_for_completed()
                    robot.set_lift_height(0.2).wait_for_completed()
                    robot.drive_straight(distance_mm(45), speed_mmps(100)).wait_for_completed()
                    robot.turn_in_place(degrees(-90)).wait_for_completed()
                    robot.drive_straight(distance_mm(-37), speed_mmps(100)).wait_for_completed()
                    robot.set_lift_height(0).wait_for_completed()
                    
                    robot.drive_straight(distance_mm(45), speed_mmps(100)).wait_for_completed()
                    robot.set_lift_height(0.2).wait_for_completed()
                    robot.drive_straight(distance_mm(45), speed_mmps(100)).wait_for_completed()
                    robot.turn_in_place(degrees(-90)).wait_for_completed()
                    robot.drive_straight(distance_mm(-40), speed_mmps(100)).wait_for_completed()
                    robot.set_lift_height(0).wait_for_completed()
                robot.set_lift_height(0.2).wait_for_completed()
                

        elif ("SUM" in text_split[i]):

            sum_num = text_split[i].replace(" ","")
            x = sum_num[sum_num.find("SUM(")+4:sum_num.find("+")]
            y = sum_num[sum_num.find("+")+1:sum_num.find(")")]
            
            text_to_draw = x + "+" + y
            text_image = make_text_image(text_to_draw)
            
            robot.say_text("How much is "+ x + "+" + y).wait_for_completed()
            
            duration_s = 3
            text_image_face = text_image
            oled_face_data = cozmo.oled_face.convert_image_to_screen_data(text_image_face)
    
            robot.display_oled_face_image(oled_face_data, duration_s * 1000)
            time.sleep(duration_s)
            
            text_ans =  main()
            text_ans_float = extract_number_word(text_ans)
            
            x = float(x)
            y = float(y)
            resp_sum = x+y
            
            
            print("Resp Falada = {}".format(text_ans_float))
            print("Resp calcuada = {}".format(resp_sum))

            
            script_results = script_results + "Answer given: " + text_ans +"\n"
            
            if (resp_sum in text_ans_float):
                
                
                robot.play_anim(name="anim_meetcozmo_celebration_02").wait_for_completed()
                robot.say_text("Congratulations!!").wait_for_completed()
                
            else:
                resp_round = round(resp_sum)
                if ((resp_round/resp_sum) == 1):
                    resp_sum = resp_round
                    
                robot.play_anim(name="anim_memorymatch_failgame_cozmo_03").wait_for_completed()
                robot.say_text("the correct answer is: {}".format(resp_sum)).wait_for_completed()
                
            
                
                
                
        elif ("SUB" in text_split[i]):

            sum_num = text_split[i].replace(" ","")
            x = sum_num[sum_num.find("SUB(")+4:sum_num.find("-")]
            y = sum_num[sum_num.find("-")+1:sum_num.find(")")]
            
            text_to_draw = x + "-" + y
            text_image = make_text_image(text_to_draw)
            
            robot.say_text("How much is "+ x + "minus" + y).wait_for_completed()
            
            duration_s = 3
            text_image_face = text_image
            oled_face_data = cozmo.oled_face.convert_image_to_screen_data(text_image_face)
    
            robot.display_oled_face_image(oled_face_data, duration_s * 1000)
            time.sleep(duration_s)
            text_ans =  main()
            text_ans_float = extract_number_word(text_ans)
            
            x = float(x)
            y = float(y)
            resp_sub = x-y
            
            
            print("Resp Falada = {}".format(text_ans_float))
            print("Resp calcuada = {}".format(resp_sub))
            
            script_results = script_results + "Answer given: " + text_ans +"\n"
            
            if (resp_sub in text_ans_float):
                
                
                robot.play_anim(name="anim_meetcozmo_celebration_02").wait_for_completed()
                robot.say_text("Congratulations!!").wait_for_completed()
                
            else:
                resp_round = round(resp_sub)
                if ((resp_round/resp_sub) == 1):
                    resp_sub = resp_round
                    
                robot.play_anim(name="anim_memorymatch_failgame_cozmo_03").wait_for_completed()
                robot.say_text("the correct answer is: {}".format(resp_sub)).wait_for_completed()
                
                
        elif ("MULT" in text_split[i]):

            sum_num = text_split[i].replace(" ","")
            x = sum_num[sum_num.find("MULT(")+5:sum_num.find("*")]
            y = sum_num[sum_num.find("*")+1:sum_num.find(")")]
            
            text_to_draw = x + "x" + y
            text_image = make_text_image(text_to_draw)
            
            robot.say_text("How much is "+ x + "times" + y).wait_for_completed()
            
            duration_s = 3
            text_image_face = text_image
            oled_face_data = cozmo.oled_face.convert_image_to_screen_data(text_image_face)
    
            robot.display_oled_face_image(oled_face_data, duration_s * 1000)
            time.sleep(duration_s)
            
            text_ans =  main()
            text_ans_float = extract_number_word(text_ans)
            
            x = float(x)
            y = float(y)
            resp_mult = x*y
            
            
            print("Resp Falada = {}".format(text_ans_float))
            print("Resp calcuada = {}".format(resp_mult))
            
            script_results = script_results + "Answer given: " + text_ans +"\n"
            
            if (resp_mult in text_ans_float):
                
                
                robot.play_anim(name="anim_meetcozmo_celebration_02").wait_for_completed()
                robot.say_text("Congratulations!!").wait_for_completed()
                
            else:
                resp_round = round(resp_mult)
                if ((resp_round/resp_mult) == 1):
                    resp_mult = resp_round
                    
                robot.play_anim(name="anim_memorymatch_failgame_cozmo_03").wait_for_completed()
                robot.say_text("the correct answer is: {}".format(resp_mult)).wait_for_completed()
                
                
                
                
        elif ("DIV" in text_split[i]):

            sum_num = text_split[i].replace(" ","")
            x = sum_num[sum_num.find("DIV(")+4:sum_num.find("/")]
            y = sum_num[sum_num.find("/")+1:sum_num.find(")")]
            
            text_to_draw = x + "÷" + y
            text_image = make_text_image(text_to_draw)
            
            robot.say_text("How much is "+ x + " divided by " + y).wait_for_completed()
            
            duration_s = 3
            text_image_face = text_image
            oled_face_data = cozmo.oled_face.convert_image_to_screen_data(text_image_face)
    
            robot.display_oled_face_image(oled_face_data, duration_s * 1000)
            time.sleep(duration_s)
            
            text_ans =  main()
            text_ans_float = extract_number_word(text_ans)
            
            x = float(x)
            y = float(y)
            resp_div = x/y
            
            
            print("Resp Falada = {}".format(text_ans_float))
            print("Resp calcuada = {}".format(resp_div))
            
            script_results = script_results + "Answer given: " + text_ans +"\n"
            
            if (resp_div in text_ans_float):
                
                
                robot.play_anim(name="anim_meetcozmo_celebration_02").wait_for_completed()
                robot.say_text("Congratulations!!").wait_for_completed()
                
            else:
                resp_round = round(resp_div)
                if ((resp_round/resp_div) == 1):
                    resp_div = resp_round
                    
                robot.play_anim(name="anim_memorymatch_failgame_cozmo_03").wait_for_completed()
                robot.say_text("the correct answer is: {}".format(resp_div)).wait_for_completed()
                  
        elif ("QUADRATIC EQUATION" in text_split[i]):
            
            
            eq_solved = False
            quadratic_eq = text_split[i].replace(" ","")
            a = quadratic_eq[quadratic_eq.find("(A=")+3:quadratic_eq.find(",B=")]
            b = quadratic_eq[quadratic_eq.find(",B=")+3:quadratic_eq.find(",C=")]
            c = quadratic_eq[quadratic_eq.find(",C=")+3:quadratic_eq.find(")")]
            
            a = float(a)
            b = float(b)
            c = float(c)
            
            if (a.is_integer() == True):
                a = int(a)
            if (b.is_integer() == True):
                b = int(b)                
            if (c.is_integer() == True):
                c = int(c)               
                
            if (b>0):
                sin_b = "+"
            else:
                sin_b = ""
            if (c>0):
                sin_c = "+"
            else:
                sin_c = ""
            
            a_str = str(a)
            b_str = str(b)
            c_str = str(c)
            
            if (a == 1):
                a_str = ""
            if (b==1):
                b_str = ""
            if (c==1):
                c_str = ""
                
                
            text_to_draw = a_str+"x^2" + sin_b + b_str + "x" + sin_c + c_str + "=0"
            text_image = make_text_image(text_to_draw)
            
            robot.say_text("Solve the equation").wait_for_completed()
            robot.say_text(a_str+"x square" + sin_b + b_str + "x" + sin_c + c_str + "=0").wait_for_completed()
           
            
            duration_s = 3
            text_image_face = text_image
            oled_face_data = cozmo.oled_face.convert_image_to_screen_data(text_image_face)
    
            robot.display_oled_face_image(oled_face_data, duration_s * 1000)
            time.sleep(duration_s)
            
            delta = math.pow(b,2) - 4*a*c
            
            if (delta < 0):
                text_recog =  main()
                robot.say_text("Are the roots real? Or imaginary?").wait_for_completed()
                text_recog =  main()
                text_recog = text_recog.upper()
                if ("REAL" in text_recog):

                    robot.play_anim(name="anim_memorymatch_failgame_cozmo_03").wait_for_completed()
                    robot.say_text("the correct answer is: Imaginary roots").wait_for_completed()                    
                elif ("IMAGINARY" in text_recog):
                    robot.play_anim(name="anim_meetcozmo_celebration_02").wait_for_completed()
                    robot.say_text("Congratulations!!").wait_for_completed() 
                    
                    
            else:
                x1 = (-b + math.sqrt(delta))/(2*a)
                x2 = (-b - math.sqrt(delta))/(2*a)
                
                x1 = round(x1,1)
                x2 = round(x2,1)
                text_recog =  main()
                robot.say_text("Are the roots real? Or imaginary?").wait_for_completed()
                text_recog =  main()
                text_recog = text_recog.upper()
                if ("REAL" in text_recog):
                    
                    robot.say_text("What is first root?").wait_for_completed()
                    text_ans1 = main()
                    text_ans1_float = extract_number_word(text_ans1)
            
            
                    if (x1 in text_ans1_float or x2 in text_ans1_float):
                
                        robot.say_text("What is second root?").wait_for_completed()
                        text_ans2 = main()
                        text_ans2_float = extract_number_word(text_ans2)
                        
                        if (x1 in text_ans1_float and x2 in text_ans2_float):
                            
                            robot.play_anim(name="anim_meetcozmo_celebration_02").wait_for_completed()
                            robot.say_text("Congratulations!!").wait_for_completed()
                            
                        elif (x1 in text_ans2_float and x2 in text_ans1_float):
                            robot.play_anim(name="anim_meetcozmo_celebration_02").wait_for_completed()
                            robot.say_text("Congratulations").wait_for_completed() 
                                
                        else:
                            robot.play_anim(name="anim_memorymatch_failgame_cozmo_03").wait_for_completed()
                            robot.say_text("the correct answer is: {}".format(x1) +" and {}".format(x2)).wait_for_completed()
                    else:
                        robot.play_anim(name="anim_memorymatch_failgame_cozmo_03").wait_for_completed()
                        robot.say_text("the correct answer is: {}".format(x1) +" and {}".format(x2)).wait_for_completed()
                                        

       
        elif ("DERIVATIVE FUNCTION" in text_split[i]):
            
            equation = text_split[i].replace(" ","")
            eq_speak = equation.replace("^"," to the ")
            eq_speak = eq_speak.replace("*","")
            eq_speak = eq_speak[eq_speak.find("(f=")+3:eq_speak.find(")for")]
            
            eq_to_draw = equation.replace("*","")
            eq_to_draw = eq_to_draw[eq_to_draw.find("(f=")+1:eq_to_draw.find(")for")]
            
            eq_py = equation.replace("^","**")
            x_value = equation[equation.find("x=")+2:len(equation)]
            x_value = float(x_value)
            eq_to_diff = eq_py[eq_py.find("(f=")+3:eq_py.find(")for")]
            
            
            eq_to_diff = sympify(eq_to_diff)
            
            x = Symbol('x')
            diff_equation = eq_to_diff.diff(x)

            
            fun_diff = lambdify(x, diff_equation, 'numpy')
            result_diff = fun_diff(x_value)

            
            robot.say_text("Find the derivative of the following equation").wait_for_completed()
            robot.say_text(eq_speak + "for x equals to {}".format(x_value)).wait_for_completed()
            
            
                  
            eq_to_draw = eq_to_draw + "\n f ' ({})".format(x_value) + " = ?"
            text_image = make_text_image(eq_to_draw)          
            duration_s = 3
            text_image_face = text_image    
            oled_face_data = cozmo.oled_face.convert_image_to_screen_data(text_image_face)
    
            robot.display_oled_face_image(oled_face_data, duration_s * 1000)
            time.sleep(duration_s)
            
            text_ans =  main()
            text_ans_float = extract_number_word(text_ans)
            
            if (result_diff in text_ans_float):
                
                robot.play_anim(name="anim_meetcozmo_celebration_02").wait_for_completed()
                robot.say_text("Congratulations!!").wait_for_completed()
                
            else:
                
                robot.play_anim(name="anim_memorymatch_failgame_cozmo_03").wait_for_completed()
                robot.say_text("the correct answer is: {}".format(result_diff)).wait_for_completed()

        
        elif ("SPEAK" in text_split[i]):

            text = text_split[i]
            text = text[text.find("(")+1:text.find(")")]
            robot.say_text(text).wait_for_completed()
      
        elif ("ASK" in text_split[i] and "END" not in text_split[i]):            
            
            text_ask = text_split[i]
            text_ask = text_ask[text_ask.find("(")+1:text_ask.find(")")]
            robot.say_text(text_ask).wait_for_completed()
    
        elif ("FORWARD" in text_split[i]):
            move = text_split[i].replace(" ", "")
            move = move.upper()
            move_dist = move[move.find("(")+1:move.find(")")]
            move_dist_float = float(move_dist)
            robot.drive_straight(distance_mm(move_dist_float), speed_mmps(100)).wait_for_completed()
       
        elif ("BACKWARD" in text_split[i]):
            move = text_split[i].replace(" ", "")
            move = move.upper()
            move_dist = move[move.find("(")+1:move.find(")")]
            move_dist_float = float(move_dist)
            robot.drive_straight(distance_mm(-move_dist_float), speed_mmps(100)).wait_for_completed()
            
        elif ("RIGHT" in text_split[i]):
            move = text_split[i].replace(" ", "")
            move = move.upper()
            move_ang = move[move.find("(")+1:move.find(")")]
            move_ang_float = float(move_ang)
            robot.turn_in_place(degrees(-move_ang_float)).wait_for_completed()
            
        elif ("LEFT" in text_split[i]):
            move = text_split[i].replace(" ", "")
            move = move.upper()
            move_ang = move[move.find("(")+1:move.find(")")]
            move_ang_float = float(move_ang)
            robot.turn_in_place(degrees(move_ang_float)).wait_for_completed()
            
        elif("PEN UP" in text_split[i]):
            robot.set_lift_height(0.2).wait_for_completed()
                        
        elif("PEN DOWN" in text_split[i]):
            robot.set_lift_height(0.0).wait_for_completed()
            
            
            
        elif("HEAD UP" in text_split[i]):
            robot.set_head_angle(cozmo.robot.MAX_HEAD_ANGLE).wait_for_completed()
                        
        elif("HEAD DOWN" in text_split[i]):
            robot.set_head_angle(cozmo.robot.MIN_HEAD_ANGLE).wait_for_completed()
                        
        elif ("SPELL" in text_split[i]):
            spell_name = text_split[i].replace(" ","")
            spell_name = spell_name[spell_name.find("(")+1:spell_name.find(")")]
            spell_name_list = list(spell_name)
            fill_name = ""
            
            for loop_sp in range(0,len(spell_name)):
                if (loop_sp == len(spell_name)-1):
                    fill_name = fill_name + "_"
                else:
                    fill_name = fill_name + "_ "
                
            fill_name_list = list(fill_name)
            var_loop = True


            loop_sp = 0
            robot.say_text("Spell the word " + spell_name).wait_for_completed()
            
            while (var_loop == True):
        
                text_image = make_text_image(fill_name)
                duration_s = 20
                text_image_face = text_image
                oled_face_data = cozmo.oled_face.convert_image_to_screen_data(text_image_face)
                robot.display_oled_face_image(oled_face_data, duration_s * 1000, in_parallel=True)
                
        
                if ("_" in fill_name):
                    
                    letter = main()
                    
                    script_results = script_results + "Answer given: " + letter +"\n"
                    
                    letter = letter.upper()
                    letter = letter.replace(" ","")
                    if ("LETTER" in letter):
                        letter = letter[letter.find("LETTER")+6:len(letter)]
                        if (len(letter) == 1):
                            if (letter != spell_name_list[loop_sp]):
                                #robot.say_text("Sorry, but the letter is wrong").wait_for_completed()
                                var_loop = False
                            
                            else:
                                x = fill_name.find('_')
                                fill_name_list[x] = letter
                                fill_name = "".join(fill_name_list)
                                loop_sp = loop_sp + 1

                        else:  
                            robot.say_text("Sorry, but i didn't understand what you said. Please, say the letter again",in_parallel=True).wait_for_completed()
                    else:
                        robot.say_text("Sorry, but i didn't understand what you said. Please, say the letter again",in_parallel=True).wait_for_completed()
            
                else:
                    var_loop = False
                    
                
                    
            
            text_image = make_text_image(fill_name)
            duration_s = 3
            text_image_face = text_image
            oled_face_data = cozmo.oled_face.convert_image_to_screen_data(text_image_face)
            robot.display_oled_face_image(oled_face_data, duration_s * 1000)
            time.sleep(duration_s)
            
            
            robot.abort_all_actions()
            
            fill_name = fill_name.replace(" ","")

            if (fill_name == spell_name):
            
                robot.play_anim(name="anim_meetcozmo_celebration_02").wait_for_completed()
                robot.say_text("Congratulations! You know how to spell the word {}".format(spell_name)).wait_for_completed()
            else:
                robot.play_anim(name="anim_memorymatch_failgame_cozmo_03").wait_for_completed()
                robot.say_text("Too bad! That is not the right letter to spell the word. Look at my screen to see how it is spelled").wait_for_completed()
                
                text_image = make_text_image(spell_name)
                duration_s = 3
                text_image_face = text_image
                oled_face_data = cozmo.oled_face.convert_image_to_screen_data(text_image_face)
                robot.display_oled_face_image(oled_face_data, duration_s * 1000)
                time.sleep(duration_s)
                
        elif ("READ" in text_split[i] and "TEXT" in text_split[i]):
            last_image = None
                                
            dir_pictures = "pictures"
            robot.camera.image_stream_enabled = True
            #robot.camera.color_image_enabled = True
            pic_filename = dir_pictures+"/picture.png"
            
            
            time.sleep(3)
            
            image = robot.world.latest_image
            while image == last_image:
                time.sleep(0.02)
                image = robot.world.latest_image
                last_image = image
            
            img_latest = robot.world.latest_image.raw_image
            plt.imshow(img_latest)
            plt.show()
            
            #img_latest = robot.camera._latest_image.raw_image
            img_convert = img_latest.convert('RGBA')
            img_convert.save(pic_filename)
        
        
            file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_pictures,"picture.png")
                
            with io.open(file_name, 'rb') as image_file:
                content = image_file.read()
        
            image = vision.types.Image(content=content)
        
            response = client_vision.text_detection(image=image)
            texts = response.text_annotations
            print('Texts:')
            if (len(texts) == 0):
                robot.say_text("I did not find any words").wait_for_completed()
            else:
                text = texts[0]
                text_to_split = text.description
                text_splitted = text_to_split.split(".")
                print(text_splitted)
                for final_text in text_splitted:
                    robot.say_text(final_text,duration_scalar=1.5).wait_for_completed()
            
                    
        elif ("FIND" in text_split[i] and "FACE" in text_split[i]):
            look_around = robot.start_behavior(cozmo.behavior.BehaviorTypes.FindFaces)
            # Try to find face
            face = None
            while True:
                if face and face.is_visible:
                    robot.set_all_backpack_lights(cozmo.lights.blue_light)
                    look_around.stop()
                    robot.play_anim(name="anim_memorymatch_solo_successgame_player_01").wait_for_completed()
                    return
                    
                else:
                    robot.set_backpack_lights_off()
                    try:
                        face = robot.world.wait_for_observed_face(timeout=30)
                    except asyncio.TimeoutError:
                        robot.say_text("Didn't find a face").wait_for_completed()
                        print("Didn't find a face.")
                        return
    

        
        elif("FIND" in text_split[i] and "FACE" not in text_split[i]):
            var_loop = True
            while (var_loop == True):
                recog_objectin_upper = text_split[i].upper()
                recog_objectin_upper = recog_objectin_upper[recog_objectin_upper.find("(")+1:recog_objectin_upper.find(")")]
                
                text_recog =  main()
                text_recog = text_recog.upper()
                if ("FORWARD" in text_recog):
                    robot.drive_straight(distance_mm(50), speed_mmps(100)).wait_for_completed()
                    
                elif ("BACKWARD" in text_recog):
                    robot.drive_straight(distance_mm(-50), speed_mmps(100)).wait_for_completed()
                    
                elif ("LEFT" in text_recog):
                    robot.turn_in_place(degrees(30)).wait_for_completed()
                    
                elif ("RIGHT" in text_recog):
                    robot.turn_in_place(degrees(-30)).wait_for_completed()
                    
                elif ("PICTURE" in text_recog):
                    
                    robot.move_head(-2)
                    time.sleep(0.5)
                    
                    num_pic = 0
                    while(num_pic<3):
                        
                        
                        #Take a picture
                        '''---------------------------------------------------------'''
                        
                        last_image = None
                        
                        dir_pictures = "pictures"
                        robot.camera.image_stream_enabled = True
                        robot.camera.color_image_enabled = True
                        pic_filename = dir_pictures+"/picture.png"
                        
                        
                        time.sleep(1)
                        
                        image = robot.world.latest_image
                        while image == last_image:
                            time.sleep(0.02)
                            image = robot.world.latest_image
                            last_image = image
                        
                        
                        #robot.say_text("Say cheese!").wait_for_completed()
                        img_latest = robot.world.latest_image.raw_image
                        plt.imshow(img_latest)
                        plt.show()
                        
                        #img_latest = robot.camera._latest_image.raw_image
                        img_convert = img_latest.convert('RGBA')
                        img_convert.save(pic_filename)
                    
      
                        # The name of the image file to annotate
                        #file_name = os.path.join(os.path.dirname(__file__),dir_pictures+"/picture.png")
                    
                        file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),dir_pictures,"picture.png")
                        
      
                        print("File name:")
                        print(file_name)
                    
                        # Loads the image into memory
                        with io.open(file_name, 'rb') as image_file:
                            content = image_file.read()
                        image = vision.types.Image(content=content)
                    
                        # Performs label detection on the image file
                        response = client_vision.label_detection(image=image)
                        labels = response.label_annotations
                        
                        list_labels = list()
                        for label in labels:
                            list_labels.append(label.description.upper())
                        print(list_labels)
                        
                            
                        if (recog_objectin_upper in list_labels):
                            robot.say_text("congratulation").wait_for_completed()
                            var_loop = False
                            num_pic = 3

                        
                        robot.move_head(1)
                        time.sleep(0.4)
                        robot.move_head(0)
                        num_pic = num_pic + 1
                        
                    if (var_loop == True):
                        robot.say_text("I did not find de object").wait_for_completed()
                        var_loop = False
                        
        elif("END" in text_split[i]):
            i = tam_list
            
    
    name_file_result = "Result_" + strftime("%d-%m-%Y_%Hh%Mm%Ss", localtime())   
    with open(os.path.join(dir_results,name_file_result + ".txt"),"w") as save_file:
        save_file.write(script_results)
        
  
if __name__ == '__main__':
    app.run()
    app.debug = True