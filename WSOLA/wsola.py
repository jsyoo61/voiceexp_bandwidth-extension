import numpy as np
import matplotlib.pyplot as plt

def wsola(wav, L_s = 190, alpha = 1 , sr=16000, packet_length = 160):

    ############ hyperparameters & parameters ###############
    # L_s : minimum frequency == 85Hz, minimum period == 1/85 == 0.01176, minimum samples = 0.01176 * sr
    # L_s == 188.23 -> 190
    threshold = 0.9 # to determine whether the packet is lost or not
    num_reference = 3
    # overlap (ms)
    overlap = 10 / 8e3 #10 frames / 8kHz
    overlap_frames = np.floor(overlap * sr).astype(int)
    num_frames = len(wav)
    # number of packets must encover all frames.
    num_packet = np.ceil(num_frames / packet_length).astype(int)

    reconstructed_wav = np.zeros(num_frames)
    packet_loss_before = False
    # print('num_frames:%s'%num_frames)

    ################### algorithm ###################

    # Separate into 2 intervals:
    # (1) before last packet (possibly fragmented packet)
    # (2) last packet
    # each interval is separated into 4 cases (A, B, C, D)

    # (1) iterate before last packet
    for packet_index in range(num_packet - 1):

        current_packet_index_start = packet_index * packet_length
        current_packet_index_end = (packet_index + 1) * packet_length
        current_packet = wav[current_packet_index_start : current_packet_index_end]

        current_packet_is_lost = find_packet_loss(current_packet, threshold)

        # 4 types of cases
        # A. No packet loss before, No packet loss now : copy
        # B. No packet loss before, Yes packet loss now : start tracking
        # C. Yes packet loss before, No packet loss now : stop tracking, wsola algorithm
        # D. Yes packet loss before, Yes packet loss now : do nothing (tracking)

        # A. No packet loss before, No packet loss now
        if packet_loss_before == False and current_packet_is_lost == False:

            # print('A')
            reconstructed_wav[current_packet_index_start : current_packet_index_end] += current_packet

        # B. No packet loss before, Yes packet loss now
        elif  packet_loss_before == False and current_packet_is_lost == True:

            # print('B')
            # keep track of contiguous packet losses
            packet_loss_before = True
            lost_packet_start = packet_index

        # C. Yes packet loss before, No packet loss now
        elif packet_loss_before == True and current_packet_is_lost == False :

            # print('C')
            lost_packet_end = packet_index - 1

            # packet loss within first n (num_reference) packets
            if lost_packet_start < num_reference :

                reference_packet_start = 0

            else:

                reference_packet_start = lost_packet_start - num_reference

            reference_packet_end = lost_packet_start - 1

            reference_packet_start_index = reference_packet_start * packet_length
            reference_packet_end_index = (reference_packet_end + 1) * packet_length

            lost_packet_end_index = (lost_packet_end + 1) * packet_length

            input_index = [reference_packet_start_index, reference_packet_end_index]
            output_index = [reference_packet_start_index, lost_packet_end_index + overlap_frames]

            # print(input_index, output_index)
            # print(output_index[0], output_index[1])
            reconstructed_wav[ output_index[0] : output_index[1] ] = wsola_algorithm(wav = reconstructed_wav, input_index = input_index, output_index = output_index, L_s = L_s, alpha = alpha)
            packet_loss_before = False
            # plt.plot(wav[output_index[0] : output_index[1]])
            # plt.show()
            # plt.plot(reconstructed_wav[ output_index[0] : output_index[1] ])
            # plt.show()

            reconstructed_wav[current_packet_index_start : current_packet_index_end] += current_packet

        # else:
        #     print('D')
        # D. Yes packet loss before, Yes packet loss now : Do nothing

    # (2) Last packet
    # 4 possible cases
    # A. No packet loss before, No packet loss now : copy
    # B. No packet loss before, Yes packet loss now : wsola algorithm (right away)
    # C. Yes packet loss before, No packet loss now : stop tracking, wsola algorithm
    # D. Yes packet loss before, Yes packet loss now : wsola algorithm

    # B, C, D all lead to wsola algorithm, with different indices.
    # So separate the indexing part, and combine the wsola algorithm part, for code efficiency

    current_packet_index_start = packet_index * packet_length
    current_packet_index_end = num_frames - 1
    current_packet = wav[current_packet_index_start : current_packet_index_end]

    current_packet_is_lost = find_packet_loss(current_packet, threshold)

    # A. No packet loss before, No packet loss now
    if packet_loss_before == False and current_packet_is_lost == False:

        reconstructed_wav[current_packet_index_start : current_packet_index_end] += current_packet

    # B, C, D
    # this separation is for code efficiency
    else:
        # B. No packet loss before, Yes packet loss now
        if  packet_loss_before == False and current_packet_is_lost == True:

            lost_packet_start = packet_index
            # lost_packet_end = packet_index

            # since the audio would be more than few miliseconds, discard the following code
            # (universal version)
            # if lost_packet_start < num_reference :
            #
            #     reference_packet_start = 0
            #
            # else:
            #
            #     reference_packet_start = lost_packet_start - num_reference

            # (simplified version)
            reference_packet_start = lost_packet_start - num_reference
            reference_packet_end = lost_packet_start - 1

            reference_packet_start_index = reference_packet_start * packet_length
            reference_packet_end_index = (reference_packet_end + 1) * packet_length

            lost_packet_end_index = num_frames - 1

            input_index = [reference_packet_start_index, reference_packet_end_index]
            output_index = [reference_packet_start_index, lost_packet_end_index]

        # C. Yes packet loss before, No packet loss now
        elif packet_loss_before == True and current_packet_is_lost == False :
            lost_packet_end = packet_index - 1

            # (simplified version) (same as above)
            reference_packet_start = lost_packet_start - num_reference
            reference_packet_end = lost_packet_start - 1

            reference_packet_start_index = reference_packet_start * packet_length
            reference_packet_end_index = (reference_packet_end + 1) * packet_length

            lost_packet_end_index = (lost_packet_end + 1) * packet_length

            input_index = [reference_packet_start_index, reference_packet_end_index]
            if lost_packet_end_index + overlap_frames > num_frames - 1 :
                output_index = [reference_packet_start_index, num_frames - 1]
            else:
                output_index = [reference_packet_start_index, lost_packet_end_index + overlap_frames]

        # D. Yes packet loss before, Yes packet loss now
        else:
            lost_packet_end = packet_index

            # (simplified version) (same as above)
            reference_packet_start = lost_packet_start - num_reference
            reference_packet_end = lost_packet_start - 1

            reference_packet_start_index = reference_packet_start * packet_length
            reference_packet_end_index = (reference_packet_end + 1) * packet_length

            lost_packet_end_index = num_frames - 1

            input_index = [reference_packet_start_index, reference_packet_end_index]
            output_index = [reference_packet_start_index, lost_packet_end_index]

        if input_index[0] < 0:
            input_index[0] = 0
            output_index[0] = 0
        # print(input_index, output_index)
        # print(output_index[0], output_index[1])
        reconstructed_wav[ output_index[0] : output_index[1] ] = wsola_algorithm(wav = reconstructed_wav, input_index = input_index, output_index = output_index, L_s = L_s, alpha = alpha)
        packet_loss_before = False

    return reconstructed_wav

def find_packet_loss(wav, threshold):

    # T = len(wav)
    # is_packet_lost = (wav == 0)
    # num_lost_packet = np.sum(is_packet_lost)
    # loss_ratio = num_lost_packet / T
    # packet_lost = loss_ratio < threshold
    #
    # return packet_lost

    return (np.sum(wav == 0)/len(wav) > threshold)

def wsola_algorithm(wav, input_index, output_index, L_s, alpha):

    ############## parameters ########################
    L_in = input_index[1] - input_index[0]
    L_out = output_index[1] - output_index[0]
    # only when first consecutive packets are lost
    if L_in == 0:
        return np.zeros(L_out)
    N = np.floor( L_out/100 - 1 ).astype(int)
    L = 2 * np.floor( L_out / (N+1) ).astype(int)
    # r: search region index, r[n] is search region for segment n+1
    r = np.zeros( N ).astype(int)
    # x: segment start index, x[n] is segment for n
    x = np.zeros( N + 1 ).astype(int)
    delta_y = L // 2
    delta_x = np.floor(delta_y / alpha).astype(int)
    r[0] = input_index [0] + np.floor( - L_s - 80 * (L_out / L_in)).astype(int)
    x[0] = r[0] + L_s // 2 - delta_x
    y = np.arange(-delta_y, delta_y * N, delta_y)
    if r[0] < 0:
        r[0] = 0
    if x[0] < 0:
        x[0] = 0

    reconstructed_wav = np.zeros(L_out)
    # print('L_in:%s, L_out:%s, N:%s, L:%s, r[0]:%s, delta_y:%s, delta_x:%s'%(L_in, L_out, N, L, r[0], delta_y, delta_x))

    ################ algorithm ###################
    # 1. Making segments
    # 1-1. find cross correlation value. Search from search region
    # 1-2. select segment start index (automatically selects segment)
    # 1-3. find next search region. Iterate

    # 1. Making segments
    segments = list()
    segments.append( wav[ x[0] : x[0] + L ] )
    for segment_index in range(1, N+1):

        # 1-1. find cross correlation value. Search from search region
        C = np.zeros(L_s)
        for C_index in range(L_s):
            # print('segment_index:%s, C_index:%s, r[segment_index - 1]:%s'%(segment_index, C_index, r[segment_index - 1]))
            # print('r[segment_index - 1] + C_index:%s, r[segment_index - 1] + C_index + delta_y:%s'%(r[segment_index - 1] + C_index, r[segment_index - 1] + C_index + delta_y))
            C[C_index] = np.sum( wav[ r[segment_index - 1] + C_index : r[segment_index - 1] + C_index + delta_y ] * segments[segment_index - 1][ -delta_y : ] )

        # 1-2. select segment start index (automatically selects segment)
        x[segment_index] = r[segment_index - 1] + C.argmax()
        segments.append( wav[ x[segment_index] : x[segment_index] + L ])

        # 1-3. find next search region. Iterate
        if segment_index != N:
            r[segment_index] = x[segment_index] + delta_x - L_s // 2

    # 2. Overlapping

    # 2-1. Find beta
    segments = np.asarray(segments)

    N_d = (x + L) - input_index[1]
    N_d[N_d < 0] = 0
    N_d[N_d > L] = L # possible?

    lambda_ = L / (L - N_d)
    sum_segments = np.sum(segments, axis = 1)
    sum_segments_prime = np.asarray( [np.sum( wav[ y[k] : y[k] + L ] ) for k in range(N+1)] )
    beta = lambda_ * sum_segments_prime / sum_segments

    segments_double_prime = np.asarray( [ beta[i] * segments[i] for i in range(len(beta)) ] )

    # plt.plot(wav[output_index[0]:output_index[1]])
    # plt.show()
    # plt.plot(segments[0])
    # plt.show()
    # plt.plot(segments[1])
    # plt.show()

    # 2-2. Overlap every other delta_y intervals, hanning window
    hanning_window = np.hanning(L)
    windowed_segments = np.asarray([segments[i] * hanning_window for i in range(N+1)])
    # segments_double_prime_fft = np.fft.fft(segments_double_prime)
    # hanning_window_fft = np.fft.fft(hanning_window)

    # windowed_segments_fft = np.asarray([segments_double_prime_fft[i] * hanning_window_fft[i] for i in range(N+1)])
    # windowed_segments = np.fft.ifft(windowed_segments_fft).astype(float)
    # print('segments_double_prime_fft:%s'%str(segments_double_prime_fft.shape))
    # print('hanning_window_fft:%s'%str(hanning_window_fft.shape))
    # print('windowed_segments_fft:%s'%str(windowed_segments_fft.shape))
    # print('windowed_segments:%s'%str(windowed_segments.shape))

    # print('windowed_segments.shape%s'%str(windowed_segments.shape))
    # overlap S[0]
    reconstructed_wav[0: delta_y] += windowed_segments[0][ -delta_y : ]
    # overlap S[1]~S[N]
    for segment_index in range(N):
        # print('segment_index:%s'%segment_index)
        # print('segment_index * delta_y:%s'%(segment_index * delta_y))
        # print('reconstructed_wav[segment_index * delta_y : segment_index * delta_y + L]:%s'%reconstructed_wav[segment_index * delta_y : segment_index * delta_y + L].shape)
        # print('windowed_segments[segment_index]:%s'%windowed_segments[segment_index].shape)
        reconstructed_wav[segment_index * delta_y : segment_index * delta_y + L] += windowed_segments[segment_index]

    return reconstructed_wav
