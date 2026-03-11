import sys
import os
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
isolpharm_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, isolpharm_dir)

from Scripts.convert_to_root import _parse_and_collect

def test_should_correctly_parse_LG_events_from_txt(tmp_path):
    """
    Tests the file parsing of a valid .txt file for the LG amplification chain.
    """
    # GIVEN: a fake .txt file with known content (2 events)
    # Header format: Tstamp_us  TrigID  ... 
    # Data format:   Tstamp_int Channel Datum
    fake_txt_content = """//
Tstamp_us TrigID
100.5 1 0 0 0 0
100 0 500
100 1 510
200.5 2 0 0 0 0
200 0 600
"""
    # tmp_path creates temporary directory for tests
    fake_file = tmp_path / "Run0_list.txt"
    fake_file.write_text(fake_txt_content)

    # WHEN: the parsing function is executed on a fake file
    # _parse_and_collect returns: (clean_name, trig_ids, trig_times, all_chan_ids, all_data_LG, all_data_HG)
    results = _parse_and_collect(str(fake_file), gain="LG")
    clean_name, trig_ids, trig_times, all_chan_ids, all_data_LG, all_data_HG = results

    # THEN: results have to corresponde to those fake data just created
    assert clean_name == "Run0"
    assert trig_ids == [1, 2] # 2 events
    assert trig_times == [100.5, 200.5]
    assert all_chan_ids == [[0, 1], [0]] # The first event has channels 0 and 1, the second only channel 0
    assert all_data_LG == [[500, 510], [600]]
    assert all_data_HG == [[], []] # HG must be empty in LG mode

def test_should_handle_empty_file(tmp_path):
    """
    Tests that an empty file is handled gracefully (no events, no errors).
    """
    # GIVEN: an empty .txt file
    empty_file = tmp_path / "EmptyRun_list.txt"
    empty_file.write_text("")

    # WHEN: the parsing function is executed on the empty file
    results = _parse_and_collect(str(empty_file), gain="LG")
    clean_name, trig_ids, trig_times, all_chan_ids, all_data_LG, all_data_HG = results

    # THEN: results should indicate no events and no data
    assert clean_name == "EmptyRun"
    assert trig_ids == []
    assert trig_times == []
    assert all_chan_ids == []
    assert all_data_LG == []
    assert all_data_HG == []

def test_should_parse_both_gains(tmp_path):
    """
    Tests that both LG and HG data are parsed correctly when gain=BOTH is specified.
    """
# GIVEN: a fake .txt file with known content for both gains
    fake_txt_content = """//
Tstamp_us TrigID
100.5 1 0 0 0 0
        100 0 500 1500
        100 1 510 1510
"""

    fake_file = tmp_path / "Run0_both.txt"
    fake_file.write_text(fake_txt_content)
    # WHEN: the parsing function is being run on data containing both LG and HG data
    results = _parse_and_collect(str(fake_file), gain="BOTH")
    clean_name, trig_ids, trig_times, all_chan_ids, all_data_LG, all_data_HG = results

    #THEN both LG and HG data should be parsed correctly
    assert clean_name == "Run0_both"
    assert trig_ids == [1]
    assert trig_times == [100.5]
    assert all_chan_ids == [[0, 1]]
    assert all_data_LG == [[500, 510]]
    assert all_data_HG == [[1500, 1510]]